# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import math
from typing import Optional, List, Tuple

import torch
import torch_npu
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    TensorEmbedding,
    AttentionMask,
    TensorParallelHead,
    get_linear,
)
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.utils.quantize.w8a8 import calc_linear_pack_type
from atb_llm.utils.data.weight_wrapper import AttnModuleNames, MlpModuleNames, WeightWrapper
from atb_llm.utils.layers import load_column_multi

from .config import ChatglmConfig


class RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states, residual


class RMSNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class RMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = RMSNorm(prefix, weights, eps)
        self.anti = RMSNormBias(f'{prefix}.module', weights, eps)



def load_qkv(config, prefix: str, weights):
    weight = weights.get_weights_col_packed_qkv_glm(config, prefix, None, False)
    bias = weights.get_weights_col_packed_qkv_glm(config, prefix, None, True)
    linear = get_linear(weight, bias, config.quantize)

    return TensorParallelColumnLinear(linear)


def load_gate_up_proj(config, prefix, weights, size):
    weight = weights.get_gate_up_glm(prefix, size)
    bias = None
    linear = get_linear(weight, bias, config.quantize)
    return TensorParallelColumnLinear(linear)


class FlashChatglmAttention(torch.nn.Module):
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

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.num_heads, base=10000.0, device="cpu").to(weights.device)

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()

        linear_names = [f'{prefix}.query_key_value']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        elif weights.quantize == "smooth_quant":
            self.pack_type = PackType.ALL_W8A8
        else:
            self.pack_type = PackType.ALL_FP

        self.query_key_value = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            bias=True,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.multi_query_group_num
        )
        self.dense = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense",
            weights=weights,
            bias=False,
        )

        self.prefix = prefix

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        is_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_seq_len,
    ):
        qkv = self.query_key_value(hidden_states)
        query, key, value = qkv.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=1)

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)

        query_embed = self.rotary_emb(query, cos, sin)
        key_embed = self.rotary_emb(key, cos, sin)

        block_num, block_size, head_num, head_size = kv_cache[0].shape
        slots_list = slots.tolist()
        for i, slot in enumerate(slots_list):
            block_index = slot // block_size
            block_offset = slot % block_size

            token_key = key_embed[i]
            token_v = value[i]
            kv_cache[0][block_index][block_offset] = token_key
            kv_cache[1][block_index][block_offset] = token_v

        # output tensor
        attn_output = torch.zeros(size=query_embed.shape, device=query_embed.device)

        # Prefill
        if is_prefill is not None:
            # flash attention
            attn_output = attention_ascend(
                query_embed,  # [n_tokens, head_num, head_size]
                key_embed,  # [n_tokens, head_num, head_size]
                value,  # [n_tokens, head_num, head_size]
                attn_output,
                is_prefill,
                max_seq_len,
                self.softmax_scale,
            )

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


class MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

        linear_names = [f'{prefix}.dense_h_to_4h']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        elif weights.quantize == "smooth_quant":
            self.pack_type = PackType.ALL_W8A8
        else:
            self.pack_type = PackType.ALL_FP

        # Fuse gate and up proj
        self.gate_up_proj = TensorParallelColumnLinear.load_gate_up(
            config,
            prefix=f"{prefix}.dense_h_to_4h",
            weights=weights,
            bias=False,
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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / \
            (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / \
            (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        emb = torch.stack((idx_theta, idx_theta), dim=-1)
        rope_cos = torch.cos(emb)
        rope_sin = torch.sin(emb)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            if dtype == torch.bfloat16:
                rope_cos = rope_cos.bfloat16()
                rope_sin = rope_sin.bfloat16()
            else:
                rope_cos = rope_cos.half()
                rope_sin = rope_sin.half()
        
        return rope_cos, rope_sin

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


class FlashDecoderLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.encoder.layers.{layer_id}"
        self.self_attention = FlashChatglmAttention(
            prefix=f"{prefix}.self_attention", config=config, weights=weights
        )
        self.mlp = MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        if self.self_attention.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.input_layernorm = RMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.layernorm_epsilon
            )
        elif self.self_attention.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.input_layernorm = RMSNormBias(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.layernorm_epsilon
            )
        else:
            self.input_layernorm = RMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.layernorm_epsilon
            )

        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.post_attention_layernorm = RMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.layernorm_epsilon,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.post_attention_layernorm = RMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.layernorm_epsilon,
            )
        else:
            self.post_attention_layernorm = RMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.layernorm_epsilon
            )

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        is_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_seq_len,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attention(
            normed_hidden_states,
            cos,
            sin,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashChatglmModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.config = config
        self.dtype = weights.dtype
        self.embed_tokens = TensorEmbedding(
            prefix="transformer.embedding.word_embeddings", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashDecoderLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix="transformer.encoder.final_layernorm", weights=weights, eps=config.layernorm_epsilon
        )

        self.gradient_checkpointing = False
        self.soc_info = NPUSocInfo()

        self.head_size = self.layers[0].self_attention.head_size
        self.num_heads = self.layers[0].self_attention.num_heads

        # for ascend init
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.kv_head_num = max(config.multi_query_group_num // self.tp_world_size, 1)
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope,
                                              device=weights.device,
                                              dtype=config.torch_dtype)
        self.cos_embed, self.sin_embed = self.rotary_pos_emb(config.seq_length)
        self.placeholder = torch.zeros(1, dtype=config.torch_dtype, device="npu")

    def init_ascend_operations(self, config: ChatglmConfig):
        self.quantize = config.quantize
        self.seq_length = config.seq_length

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("glm_v2_6b_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("glm_v2_6b_DecoderModel")

        self.weight_flag = False
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size

        self.acl_encoder_operation_inputs = [None] * 12
        self.acl_decoder_operation_inputs = [None] * 12
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64)
        # self.lm_head_weight = None

        self.attn_mask = AttentionMask.static(config.seq_length)

    def weight_format_cast(self, weight):
        if not self.soc_info.need_nz:
            return weight

        torch_npu.npu_format_cast_(weight, 29)
        return weight

    def init_ascend_weight(self):
        self.ascend_weight, self.linear_type, self.pack_quant_config = self.get_weights()
        pre_scale = []
        for layer_id in range(1, self.config.num_layers + 1):
            pre_scale.append(layer_id / (math.sqrt(self.config.kv_channels) * layer_id))
        post_scale = [1.0] * self.config.num_layers
        coder_param = {
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "isEmbeddingParallel": False,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "rmsNormEps": self.config.layernorm_epsilon,
            "numAttentionHeadsPerRank": self.config.num_attention_heads // self.tp_world_size,
            "hiddenSizePerAttentionHead": self.config.hidden_size // self.config.num_attention_heads,
            "numHiddenLayers": self.config.num_layers,
            "numKeyValueHeadsPerRank": max(self.config.multi_query_group_num // self.tp_world_size, 1),
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "isLmHeadParallel": True,
            "packQuantType": self.pack_quant_config,
            "linearQuantType": self.linear_type,
            "tokenOffset": [0],
            "seqLen": [1],
            "preScale": pre_scale,
            "postScale": post_scale,
        }
        encoder_param = {**coder_param, "isPrefill": True}
        decoder_param = {**coder_param, "isPrefill": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

        self.lm_head_indices_fake = self.lm_head_indices_fake.to(self.state_dict()["embed_tokens.weight"].device)

    def get_weights(self):
        quant_type = []
        attn_module_names = AttnModuleNames(
            norm_name='input_layernorm',
            pack_name='self_attention.query_key_value',
            o_name='self_attention.dense'
        )
        mlp_module_names = MlpModuleNames(
            norm_name='post_attention_layernorm',
            pack_name='mlp.gate_up_proj',
            down_name='mlp.down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_module_names, mlp_module_names)
        weight_wrapper.register_embedding(self.state_dict(), 'embed_tokens')
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer_dict = layer.state_dict()
            weight_wrapper.register_layer(layer_dict,
                                          layer.self_attention.pack_type,
                                          layer.mlp.pack_type,
                                          self.quantize)
            quant_type.append([layer.self_attention.pack_type.value, layer.mlp.pack_type.value])
            if self.soc_info.need_nz:
                del layer.self_attention
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.state_dict(), 'norm')
        weight_wrapper.weights.append(self.weight_format_cast(self.lm_head_weight))
        return weight_wrapper.weights, weight_wrapper.linear_type, quant_type

    def init_ascend_kvcache(self, kv_cache):
        kv_cache_exist = self.ascend_kcache_id and self.ascend_vcache_id
        if not kv_cache_exist or \
            self.ascend_kcache_id != id(kv_cache[0][0]) or \
            self.ascend_vcache_id != id(kv_cache[0][1]):
            k_caches, v_caches = map(list, zip(*kv_cache))
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None
    ):
        cos_embed, sin_embed = self.cos_embed[position_ids.long()], self.sin_embed[position_ids.long()]

        if is_prefill is not None:  # prefill
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            
            if self.soc_info.need_nz:
                self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
                self.transdata_param = json.dumps({})
                self.transdata_operation.set_param(self.transdata_param)

                pad_maxs = math.ceil(self.seq_length / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.attn_mask.get_attn_mask(self.seq_length, kv_cache[0][0].dtype, kv_cache[0][0].device)
            self.acl_param_encoder = json.dumps({
                "seqLen" : input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[2] = cos_embed
            self.acl_encoder_operation_inputs[3] = sin_embed
            self.acl_encoder_operation_inputs[4] = atten_mask
            self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = self.placeholder
            self.acl_encoder_operation_inputs[8] = self.placeholder
            self.acl_encoder_operation_inputs[9] = self.placeholder
            self.acl_encoder_operation_inputs[10] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[11] = lm_head_indices.to(torch.int64)

            return self.acl_encoder_operation_inputs, self.acl_param_encoder
        else:
            atten_mask = torch.tensor([1], device=input_ids.device, dtype=kv_cache[0][0].dtype)
            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = cos_embed
            self.acl_decoder_operation_inputs[3] = sin_embed
            self.acl_decoder_operation_inputs[4] = atten_mask
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = self.placeholder
            self.acl_decoder_operation_inputs[8] = self.placeholder
            self.acl_decoder_operation_inputs[9] = self.placeholder
            self.acl_decoder_operation_inputs[10] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[11] = lm_head_indices.to(torch.int64)

            return self.acl_decoder_operation_inputs, self.acl_param_decoder

    def execute_ascend_operator(self,
                                input_ids: torch.Tensor,
                                position_ids: torch.Tensor,
                                is_prefill: bool,
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_seq_len: int,
                                lm_head_indices: Optional[torch.Tensor] = None):
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len, lm_head_indices)

        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)

        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]

        return acl_hidden_state


    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_ascend_kvcache(kv_cache)
        hidden_states_acl = self.execute_ascend_operator(input_ids, position_ids, is_prefill, kv_cache,
                                                     block_tables, slots, input_lengths, max_seq_len, lm_head_indices)

        return hidden_states_acl


class FlashChatglmForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        load_atb_speed()

        self.transformer = FlashChatglmModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["transformer.output_layer"],
            weights=weights,
            head_size=1,
            lm_head=True
        )

        # for ascend
        self.tp_world_size = self.transformer.tp_world_size
        self.soc_info = self.transformer.soc_info
        self.head_size = self.transformer.head_size
        self.num_attention_heads = self.transformer.num_heads // self.tp_world_size
        self.num_key_value_heads = self.transformer.kv_head_num
        self.num_layers = self.transformer.num_layers

        self.lm_head_weight = None

    def forward(
        self,
        input_ids: torch.Tensor,  # input id, 拉平的
        position_ids: torch.Tensor,  #
        is_prefill: bool,  # 是否prefill阶段
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
        block_tables: torch.Tensor,  # 每个requests 所有的block tables
        slots: torch.Tensor,  # 每个requests 所有的slots
        input_lengths: torch.Tensor,  # 每个 request的k/v长度
        max_seq_len: int,  # 最长的request长度
        lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
    ) -> torch.Tensor:
        if self.lm_head_weight is None:
            self.lm_head_weight = self.state_dict()["lm_head.linear.weight"]
            self.transformer.lm_head_weight = self.lm_head_weight
        
        logits = self.transformer(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices
        )

        return logits
