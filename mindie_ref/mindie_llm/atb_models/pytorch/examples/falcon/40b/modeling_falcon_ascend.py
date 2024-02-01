# coding=utf-8
# Copyright 2023 the Falcon authors and HuggingFace Inc. team.  All rights reserved.
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
"""PyTorch Falcon model."""

import os
import json
import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch_npu
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers import FalconConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# ======================== load so ========================
ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
if ATB_SPEED_HOME_PATH is None:
    raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(ATB_SPEED_HOME_PATH, "lib/libatb_speed_torch.so")
torch.classes.load_library(LIB_PATH)
# =========================================================


# NOTE(Hesslow): Unfortunately we did not fuse matmul and bias during training, this means that there's one additional quantization to bfloat16 between the operations.
# In order not to degrade the quality of our HF-port, we keep these characteristics in the final model.
class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_states = input @ self.weight.T
        if self.bias is None:
            return hidden_states
        return hidden_states + self.bias


# rotary pos emb helpers (torch.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class FalconRotaryEmbedding(nn.Module):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with `[batch_size,
    n_heads_per_partition, seq_len, head_dim]` (e.g. MinGPTAttention format).
    """

    def __init__(self, head_dim: int, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).double() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = -1
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def cos_sin(self, seq_len: int, past_key_values_length: int, device="cpu", dtype=torch.bfloat16) -> torch.Tensor:
        total_length = seq_len + past_key_values_length
        if total_length > self.seq_len_cached:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            self.seq_len_cached = total_length
            t = torch.arange(total_length, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)

        return (
            self.cos_cached[:, past_key_values_length : seq_len + past_key_values_length],
            self.sin_cached[:, past_key_values_length : seq_len + past_key_values_length],
        )

    def forward(self, query, key, past_key_values_length=0):
        batch, seq_len, head_dim = query.shape
        cos, sin = self.cos_sin(seq_len, past_key_values_length, query.device, query.dtype)
        return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class FalconLinearScalingRotaryEmbedding(FalconRotaryEmbedding):
    """FalconRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, head_dim: int, base=10000, max_position_embeddings=2048, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(head_dim, base, max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).to(dtype)
        # This line is the only difference from FalconRotaryEmbedding._set_cos_sin_cache
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        if dtype in [torch.float16, torch.bfloat16]:
            emb = emb.float()

        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

        self.cos_cached = self.cos_cached.type(dtype)
        self.sin_cached = self.sin_cached.type(dtype)


class FalconDynamicNTKScalingRotaryEmbedding(FalconRotaryEmbedding):
    """
    FalconRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self, head_dim: int, base=10000, max_position_embeddings=2048, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(head_dim, base, max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.seq_len_cached = seq_len

        # This if block is the only difference from FalconRotaryEmbedding._set_cos_sin_cache
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.head_dim / (self.head_dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device).to(dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        if dtype in [torch.float16, torch.bfloat16]:
            emb = emb.float()

        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

        self.cos_cached = self.cos_cached.type(dtype)
        self.sin_cached = self.sin_cached.type(dtype)


# Copied from transformers.models.bloom.modeling_bloom.dropout_add
def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class FalconAttention(nn.Module):
    def __init__(self, config: FalconConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads # 8192 / 128 = 64
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.is_causal = True
        self.world_size = 1 if not hasattr(config, 'world_size') else config.world_size
        self.num_heads =  self.num_heads // self.world_size # 128 / 4 = 32
        self.maybe_rotary = self._init_rope() if config.rotary else lambda q, k, t, p: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if config.new_decoder_architecture:
            # (8*2 + 128) * 64
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        # [+] FalconLinear(8192, 2304, False)
        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim // self.world_size, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(self.hidden_size // self.world_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1
        self.num_kv_heads = self.num_kv_heads // self.world_size

    def _init_rope(self):
        if self.config.rope_scaling is None:
            rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                base=self.config.rope_theta
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        return rotary_emb


class FalconMLP(nn.Module):
    def __init__(self, config: FalconConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.world_size = 1 if not hasattr(config, 'world_size') else config.world_size
        self.dense_h_to_4h = FalconLinear(hidden_size, 4 * hidden_size // self.world_size, bias=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = FalconLinear(4 * hidden_size // self.world_size, hidden_size, bias=config.bias)
        self.hidden_dropout = config.hidden_dropout


class FalconDecoderLayer(nn.Module):
    def __init__(self, config: FalconConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.self_attention = FalconAttention(config)
        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if config.new_decoder_architecture:
            # The layer norm before self-attention
            self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            # The layer norm before the MLP
            self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            if not config.parallel_attn:
                self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)



FALCON_START_DOCSTRING = r"""
"""

FALCON_INPUTS_DOCSTRING = r"""
"""


class FalconPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FalconConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FalconDecoderLayer"]
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear) or isinstance(module, FalconLinear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _convert_cache_to_standard_format(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, kv_length, head_dim = past_key_value[0][0].shape
        # [batch_size * self.num_heads, kv_length, head_dim] -> [batch_size, num_heads, kv_length, head_dim]
        # Note that don't want to use self.num_attention_heads because the number of heads may vary depending
        # on whether we use multi_query attention.
        num_heads = batch_size_times_num_heads // batch_size
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, kv_length, head_dim),
                layer_past[1].view(batch_size, num_heads, kv_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_rw_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_heads, kv_length, head_dim = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # [batch_size, num_heads, kv_length, head_dim] -> [batch_size * num_heads, kv_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, kv_length, head_dim),
                layer_past[1].view(batch_size_times_num_heads, kv_length, head_dim),
            )
            for layer_past in past_key_value
        )


@add_start_docstrings(
    "The bare Falcon Model transformer outputting raw hidden-states without any specific head on top.",
    FALCON_START_DOCSTRING,
)
class FalconModel(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        self.world_size = 1 if not hasattr(config, 'world_size') else config.world_size
        self.num_heads = self.num_heads // self.world_size

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings


def get_distributed_info():
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        rankSize = torch.distributed.get_world_size()
    else:
        rank = 0
        rankSize = 1
    return rank, rankSize


@add_start_docstrings(
    "The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).",
    FALCON_START_DOCSTRING,
)
class FalconForCausalLM(FalconPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.transformer = FalconModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rank, self.world_size = get_distributed_info()
        # Initialize weights and apply final processing
        self.post_init()

        # -------------------------atb_speed------------------------------
        self.num_hidden_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden_size // self.num_heads
        self.total_seqlen = 2048

        self.rope = FalconRotaryEmbedding(config.head_dim)

        self.cos_cache, self.sin_cache = self.rope.cos_cached, self.rope.sin_cached
        self.cache_k = None
        self.cache_v = None
        self.attention_mask_max = None
        self.is_set_weight = False

        self.num_heads = self.num_heads // self.world_size
        self.num_kv_heads = config.num_kv_heads // self.world_size

        self.inputs_acl = [None] * (10 + self.num_hidden_layers)
        for i in range(self.num_hidden_layers):
            self.inputs_acl[i + 10] = torch.tensor([i], dtype=torch.int32).npu()


        self.param = json.dumps({
            "headNum": self.num_heads,
            "hiddenSize": config.hidden_size,
            "kvHeadNum": self.num_kv_heads,
            "layerNum": self.num_hidden_layers,
            "headDim": self.head_dim,
            "rank": self.rank,
            "rankSize": self.world_size,
            "axis": 0,
            "rotaryCoeff": 2,
            "qScale": 1.0 / math.sqrt(self.head_dim),
            "qkScale": 1.0,
            "layerNormEps": config.layer_norm_epsilon,
            "model": "falcon_40b",
        })
        self.falcon_model = torch.classes.ModelTorch.ModelTorch("falcon_40b_model")
        self.falcon_model.set_param(self.param)


    def _get_weights_float(self):
        weights = [self.transformer.word_embeddings.weight]
        for block in self.transformer.h:
            keys = [
                    'ln_attn.weight',
                    'ln_attn.bias',
                    'ln_mlp.weight',
                    'ln_mlp.bias',
                    'self_attention.query_key_value.weight',
                    'self_attention.dense.weight',
                    'mlp.dense_h_to_4h.weight',
                    'mlp.dense_4h_to_h.weight',
                ]
            weights.extend([block.state_dict()[key] for key in keys])
        weights.append(self.transformer.ln_f.weight)
        weights.append(self.transformer.ln_f.bias)
        weights.append(self.lm_head.weight)
        return weights
  
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            # past_length = past_key_values[0][0].shape[2]
            past_length = self.seq_len

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # Note: versions of Falcon with alibi do not use position_ids. It is used with RoPE.
        if not self.transformer.use_alibi and attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint="Rocketknight1/falcon-rw-1b",
        output_type=CausalLMOutputWithCrossAttentions,
        config_class="FalconConfig",
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if not self.is_set_weight:
            self.falcon_model.set_weight(self._get_weights_float())
            print("[+] Set falcon model weights success")
            self.is_set_weight = True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ---------------------acl------------------

        device = input_ids.device
        batch_size, seq_length = input_ids.shape

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values:
            past_key_values_length = self.seq_len
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        else:
            attention_mask = attention_mask.to(device)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.word_embeddings(input_ids)

        causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )


        if past_key_values:
            self.seq_len += 1 # pred next location
            seqlen = torch.tensor([1] * batch_size, dtype=torch.int32, device=device)
            token_offset = torch.tensor([self.seq_len] * batch_size, dtype=torch.int32, device=device)
            position_id = torch.tensor([[self.seq_len - 1]] * batch_size, dtype=torch.int64, device=device) # -1 is current location
            self.attention_mask_max[:, :1, :self.seq_len] = causal_mask[:, 0, :, :]
        else:
            self.total_seqlen = getattr(self, "total_seq_len", 4096)
            self.seq_len = seq_length
            seqlen = torch.tensor([self.seq_len] * batch_size, dtype=torch.int32, device=device)
            token_offset = torch.tensor([self.seq_len] * batch_size, dtype=torch.int32, device=device)
            position_id = torch.tensor([list(range(self.seq_len))] * batch_size, dtype=torch.int64, device=device)
            if self.attention_mask_max is None or batch_size != self.attention_mask_max.shape[0] or self.total_seqlen != self.attention_mask_max.shape[-2]:
                self.attention_mask_max = torch.full((batch_size, self.total_seqlen, self.total_seqlen), -10000, dtype=torch.float16, device=device)
            self.attention_mask_max[:, :self.seq_len, :self.seq_len] = causal_mask[:, 0, :, :]

            if self.cache_k is None or batch_size != self.cache_k.shape[1] or self.total_seqlen != self.cache_k.shape[-2]:
                self.cache_k = torch.zeros(self.num_hidden_layers, batch_size, self.total_seqlen, self.head_dim * self.num_heads, dtype=torch.float16, device=device)
                self.cache_v = torch.zeros(self.num_hidden_layers, batch_size, self.total_seqlen, self.head_dim * self.num_heads, dtype=torch.float16, device=device)
                self.rope.cos_sin(0, self.total_seqlen, device=device, dtype=torch.float16)
        
        param_seqlen = seqlen.cpu().tolist()
        param_token_offset = token_offset.cpu().tolist()
        run_param = json.dumps({"tokenOffset": param_token_offset, "seqLen": param_seqlen})
        rand_tensor = torch.zeros((1, 1), device=device, dtype=torch.float16)
        self.inputs_acl[:10] = [input_ids, position_id, self.rope.cos_cached.squeeze(dim=0), self.rope.sin_cached.squeeze(dim=0), self.attention_mask_max, self.cache_k, self.cache_v, token_offset, seqlen, rand_tensor]

        out = self.falcon_model.execute(self.inputs_acl, run_param)
        lm_logits = out[0]

        loss = None

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=((None), ),
            hidden_states=None,
            attentions=None,
        )


    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past
        )
        return reordered_past
