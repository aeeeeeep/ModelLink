# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
import os
from typing import Optional, List, Tuple

import torch
import torch.distributed
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from loguru import logger

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorEmbedding,
    paged_attn,
    flash_attn,
    reshape_and_cache,
    get_linear
)



class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        hidden_act="silu",
        prefix_projection=False,
        intermediate_size=11008,
        **kwargs
    ):

        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residull_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.intermediate_size = intermediate_size
        self.rotary_half = True
        self.max_position_embeddings = self.seq_length
        self.max_position_embeddings_for_npu = int(os.environ.get("MAX_SEQ_LEN", default=2048))
        self.left_padding_mask = True
        self.hidden_act = hidden_act
        setattr(self, "num_hidden_layers", self.num_layers)
        setattr(self, "rms_norm_eps", self.layernorm_epsilon)
        setattr(self, "num_key_value_heads", self.multi_query_group_num)
        super().__init__(**kwargs)


class ChatGLMRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states, residual


class ChatGLMMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()


        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]
        self.activation_func = swiglu

            # Fuse gate and up proj
        self.gate_up_proj = load_gate_up_proj(
            config,
            prefix=f"{prefix}.dense_h_to_4h",
            weights=weights,
            size=2
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense_4h_to_h",
            weights=weights,
            bias=False,
        )
        
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashChatGLMAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu", rotary_half=config.rotary_half).to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5

        # can support self.num_heads % weights.process_group.size() != 0
        if (config.num_attention_heads != config.num_key_value_heads
                and (self.num_heads % weights.process_group.size() != 0)):
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )

        self.num_heads = (self.num_heads + weights.process_group.size() - 1) // weights.process_group.size()
        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_heads

        self.query_key_value = load_qkv(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights
        )
        
        self.dense = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense",
            weights=weights,
            bias=False,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        self.prefix = prefix

    def forward(
            self,
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        qkv = self.query_key_value(hidden_states)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)

        reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            paged_attn(
                attn_output,
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


class FlashChatGLMLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.encoder.layers.{layer_id}"
        self.self_attention = FlashChatGLMAttention(
            prefix=f"{prefix}.self_attention", config=config, weights=weights
        )
        self.mlp = ChatGLMMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = ChatGLMRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = ChatGLMRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
            self,
            hidden_states,
            residual,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashChatGLMModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashChatGLMLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = ChatGLMRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            cu_seqlen_prefill: Optional[torch.Tensor],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_s: int,
            lm_head_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
             ee eddi,    
              cc,   
                sin,
                cu_sqllen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class ChatGLMModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorEmbedding(
            prefix="transformer.embedding.word_embeddings", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashChatGLMLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = ChatGLMRMSNorm(
            prefix="transformer.encoder.final_layernorm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attention.head_size
        self.num_heads = self.layers[0].self_attention.num_heads

        self.num_key_value_heads = self.layers[0].self_attention.num_key_value_heads


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            max_s: Optional[int] = None,
            lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        for _, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

def load_qkv(config, prefix: str, weights):
    hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads
    num_attention_heads_per_partition = config.num_attention_heads
    num_multi_query_groups_per_partition = config.multi_query_group_num
    q_size = hidden_size_per_attention_head * num_attention_heads_per_partition
    kv_size = hidden_size_per_attention_head * num_multi_query_groups_per_partition

    weight = weights.get_weights_col_packed_qkv_glm(prefix, None, q_size, kv_size, False)
    bias = weights.get_weights_col_packed_qkv_glm(prefix, None, q_size, kv_size, True)
    linear = get_linear(weight, bias, config.quantize)

    return TensorParallelColumnLinear(linear)


def load_gate_up_proj(config, prefix, weights, size):
    weight = weights.get_gate_up_glm(prefix, size)
    bias = None
    linear = get_linear(weight, bias, config.quantize)
    return TensorParallelColumnLinear(linear)
