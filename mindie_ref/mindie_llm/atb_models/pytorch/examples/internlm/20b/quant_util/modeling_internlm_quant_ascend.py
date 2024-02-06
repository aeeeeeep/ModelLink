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
""" PyTorch InternLM model."""
import math
from typing import List, Optional, Tuple, Union
import threading, queue

import torch
import torch_npu
import os
import json
import numpy as np
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from contextlib import contextmanager

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.streamers import BaseStreamer
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_internlm import InternLMConfig
from functools import wraps

def load_acl_transformer():
    acl_transformer_home_path = os.getenv("ATB_SPEED_HOME_PATH", "")
    if not acl_transformer_home_path or not os.path.exists(acl_transformer_home_path):
        raise RuntimeError("env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
    lib_path = os.path.join(acl_transformer_home_path, "lib/libatb_speed_torch.so")
    torch.classes.load_library(lib_path)

def load_quant_antioutlier_weight_path():
    antioutlier_weight_path = os.getenv("INTERNLM_ANTIOUTLIER_WEIGHT_PATH", "")
    if not antioutlier_weight_path or not os.path.exists(antioutlier_weight_path):
        raise RuntimeError("env INTERNLM_ANTIOUTLIER_WEIGHT_PATH not exist, source set_quant_env.sh")

    quant_weight_path = os.getenv("INTERNLM_QUANT_WEIGHT_PATH", "")
    if not quant_weight_path or not os.path.exists(quant_weight_path):
        raise RuntimeError("env INTERNLM_QUANT_WEIGHT_PATH not exist, source set_quant_env.sh")
    return antioutlier_weight_path, quant_weight_path

load_acl_transformer()

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "InternLMConfig"

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class InternLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        InternLMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class InternLMRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        import platform
        if platform.machine() == "aarch64":
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).double().to(device) / dim))
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class AscendRotaryEmbedding(InternLMRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__(dim, max_position_embeddings, base, device)
        self.cos_cached = self.cos_cached.squeeze(1).squeeze(0)
        self.sin_cached = self.sin_cached.squeeze(1).squeeze(0)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            super().forward(x, seq_len)
            self.cos_cached = self.cos_cached.squeeze(1).squeeze(0)
            self.sin_cached = self.sin_cached.squeeze(1).squeeze(0)
        if x.device != self.cos_cached.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternLMMLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class InternLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternLMConfig):
        super().__init__()
        self.config = config
        self.world_size = 1
        self.world_size = torch.distributed.get_world_size()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.num_heads = self.num_heads // self.world_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        self.rotary_emb = InternLMRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class InternLMDecoderLayer(nn.Module):
    def __init__(self, config: InternLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = InternLMAttention(config=config)
        self.world_size = 1
        self.world_size = torch.distributed.get_world_size()
        self.mlp = InternLMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size // self.world_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = InternLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = InternLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


INTERNLM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`InternLMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare InternLM Model outputting raw hidden-states without any specific head on top.",
    INTERNLM_START_DOCSTRING,
)
class InternLMPreTrainedModel(PreTrainedModel):
    config_class = InternLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLMDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternLMModel):
            module.gradient_checkpointing = value


INTERNLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class KVAttentionManager:
    def __init__(self, config: InternLMConfig, batch_size, rankSize):
        self.is_full = True
        self.batch_size = batch_size
        self.num_layers = config.num_hidden_layers
        self.world_size = 1
        self.world_size = torch.distributed.get_world_size()

        self.hidden_size = config.hidden_size // self.world_size
        self.max_sequence_length = config.max_position_embeddings

        self.k_cache_input = torch.zeros(self.num_layers,
                                         batch_size,  # batch
                                         self.max_sequence_length,
                                         self.hidden_size,
                                         device="cpu").npu().half()
        self.v_cache_input = torch.zeros(self.num_layers,
                                         batch_size,  # batch
                                         self.max_sequence_length,
                                         self.hidden_size,
                                         device="cpu").npu().half()
        self.token_offset = 1
        self.attention_mask_max_full = torch.zeros(
            (self.batch_size, self.max_sequence_length, self.max_sequence_length), dtype=torch.half).npu()
        self.attention_mask_max_inc = torch.zeros(
            (self.batch_size, self.max_sequence_length, self.max_sequence_length), dtype=torch.half).npu()
        self.ori_len_list = []
        self.min_cache = None

    def init_attention_mask(self):
        self.attention_mask_max_full.zero_()
        self.attention_mask_max_inc.zero_()

    def init_seq_len_and_token_offset(self, seq_len):
        self.token_offset = seq_len
        self.seq_len_list_full = [self.token_offset] * self.batch_size
        self.seq_len_tensor_full = torch.full((self.batch_size,), self.token_offset, dtype=torch.int32).npu()
        self.seq_len_list_inc = [1] * self.batch_size
        self.seq_len_tensor_inc = torch.full((self.batch_size,), 1, dtype=torch.int32).npu()
        self.token_offset_tensor = torch.full((self.batch_size,), self.token_offset, dtype=torch.int32).npu()

    @property
    def seq_len_list(self):
        if self.is_full:
            return self.seq_len_list_full
        return self.seq_len_list_inc

    @property
    def seq_len_tensor(self):
        if self.is_full:
            return self.seq_len_tensor_full
        return self.seq_len_tensor_inc

    @property
    def token_offset_list(self):
        return [self.token_offset] * self.batch_size

    def get_attention_mask(self, attention_mask=None):
        if not self.is_full:
            return self.attention_mask_max_inc
        else:
            for i in range(self.batch_size):
                self.attention_mask_max_full[i][:self.token_offset, :self.token_offset] = attention_mask[i]
                ori_len = self.ori_len_list[i].item()
                # 左padding
                self.attention_mask_max_inc[i][:, :self.token_offset - ori_len] = self.min_cache[:, :self.token_offset - ori_len]
                # 右padding
                # self.attention_mask_max_inc[i][:, ori_len:self.token_offset] = self.min_cache[:, ori_len:self.token_offset]
            return self.attention_mask_max_full


def load_anti_outlier_state_dict(anti_weight_path):
    antiOutlier_dict = {}
    with open(os.path.join(anti_weight_path, 'pytorch_model.bin.index.json')) as f:
        pytorch_model_index = json.load(f)
    weight_map = pytorch_model_index.get('weight_map')
    weight_files = list(set(weight_map.values()))
    for weight_file in weight_files:
        weight_tensor_dict = torch.load(os.path.join(anti_weight_path, weight_file), map_location='cpu')
        antiOutlier_dict.update(weight_tensor_dict)
    return antiOutlier_dict


def padding_descale(x):
    """
    910B上deq换算

    :param x: deq
    :return: result
    """
    zeros = torch.zeros(x.shape).npu()
    result = torch.cat((x.unsqueeze(1).npu(), zeros.unsqueeze(1).npu()), dim=1).view(-1).npu()
    return result


@add_start_docstrings(
    "The bare InternLM Model outputting raw hidden-states without any specific head on top.",
    INTERNLM_START_DOCSTRING,
)
class InternLMModel(InternLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLMDecoderLayer`]

    Args:
        config: InternLMConfig
    """
    _auto_class = "AutoModel"

    def __init__(self, config: InternLMConfig):
        super().__init__(config)
        self.rank = torch.distributed.get_rank()
        self.rankSize = torch.distributed.get_world_size()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([InternLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = InternLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_id_list = [torch.tensor([i], dtype=torch.int32).npu() for i in range(config.num_hidden_layers)]
        antioutlier_weight_path, quant_weight_path = load_quant_antioutlier_weight_path()
        self.weight_path = os.path.join(quant_weight_path, str(self.rank))

        self.input_scale_dict = np.load(os.path.join(self.weight_path, "input_scale.npy"), allow_pickle=True).item()
        self.input_offset_dict = np.load(os.path.join(self.weight_path, "input_offset.npy"), allow_pickle=True).item()
        self.antiOutlier_dict = load_anti_outlier_state_dict(os.path.join(antioutlier_weight_path, "part_model",
                                                                          str(self.rank)))

        self.gradient_checkpointing = False
        self.float_layers = [0, 1, 3, 7, 10] # 回退层

        # Initialize weights and apply final processing
        self.post_init()

        # for ascend init
        self.init_ascend_operations(config)

    def init_ascend_operations(self, config: InternLMConfig):
        q_proj_input_scale = []
        q_proj_input_offset = []
        k_proj_input_scale = []
        k_proj_input_offset = []
        v_proj_input_scale = []
        v_proj_input_offset = []
        o_proj_input_scale = []
        o_proj_input_offset = []
        gate_proj_input_scale = []
        gate_proj_input_offset = []
        down_proj_input_scale = []
        down_proj_input_offset = []

        for layer_index in range(config.num_hidden_layers):
            if layer_index in self.float_layers:
                # 回退层
                q_proj_input_scale.append(float(0))
                q_proj_input_offset.append(float(0))
                k_proj_input_scale.append(float(0))
                k_proj_input_offset.append(float(0))
                v_proj_input_scale.append(float(0))
                v_proj_input_offset.append(float(0))
                o_proj_input_scale.append(float(0))
                o_proj_input_offset.append(float(0))
                gate_proj_input_scale.append(float(0))
                gate_proj_input_offset.append(float(0))
                down_proj_input_scale.append(float(0))
                down_proj_input_offset.append(float(0))
            else:
                # 量化层
                q_proj_name = "model.layers.{}.self_attn.q_proj".format(layer_index)
                k_proj_name = "model.layers.{}.self_attn.k_proj".format(layer_index)
                v_proj_name = "model.layers.{}.self_attn.v_proj".format(layer_index)
                o_proj_name = "model.layers.{}.self_attn.o_proj".format(layer_index)
                gate_proj_name = "model.layers.{}.mlp.gate_proj".format(layer_index)
                down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)

                q_proj_input_scale.append(float(1 / self.input_scale_dict[q_proj_name]))
                q_proj_input_offset.append(int(self.input_offset_dict[q_proj_name]))
                k_proj_input_scale.append(float(1 / self.input_scale_dict[k_proj_name]))
                k_proj_input_offset.append(int(self.input_offset_dict[k_proj_name]))
                v_proj_input_scale.append(float(1 / self.input_scale_dict[v_proj_name]))
                v_proj_input_offset.append(int(self.input_offset_dict[v_proj_name]))
                o_proj_input_scale.append(float(1 / self.input_scale_dict[o_proj_name]))
                o_proj_input_offset.append(int(self.input_offset_dict[o_proj_name]))
                gate_proj_input_scale.append(float(1 / self.input_scale_dict[gate_proj_name]))
                gate_proj_input_offset.append(int(self.input_offset_dict[gate_proj_name]))
                down_proj_input_scale.append(float(1 / self.input_scale_dict[down_proj_name]))
                down_proj_input_offset.append(int(self.input_offset_dict[down_proj_name]))

        head_size = config.hidden_size // config.num_attention_heads
        self.head_num = config.num_attention_heads // self.rankSize
        self.acl_param = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": config.num_attention_heads // self.rankSize,
            "dk": head_size,
            "layerNum": config.num_hidden_layers,
            "rank": self.rank,
            "rankSize": self.rankSize,
            "q_projInputScale": q_proj_input_scale,
            "q_projInputOffset": q_proj_input_offset,
            "k_projInputScale": k_proj_input_scale,
            "k_projInputOffset": k_proj_input_offset,
            "v_projInputScale": v_proj_input_scale,
            "v_projInputOffset": v_proj_input_offset,
            "o_projInputScale": o_proj_input_scale,
            "o_projInputOffset": o_proj_input_offset,
            "gate_projInputScale": gate_proj_input_scale,
            "gate_projInputOffset": gate_proj_input_offset,
            "down_projInputScale": down_proj_input_scale,
            "down_projInputOffset": down_proj_input_offset
        })
        self.max_position_embeddings = config.max_position_embeddings
        self.acl_fa_operation = torch.classes.ModelTorch.ModelTorch("internlm_20b_flash_attention_quant_model")
        
        self.acl_fa_operation.set_param(self.acl_param)

        self.ascend_rotary_embedding = AscendRotaryEmbedding(
            config.hidden_size // config.num_attention_heads, max_position_embeddings=self.max_position_embeddings)

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.weights = []
        self.ascend_weight = []
        self.lm_head_weight = None
        self.batch_size = 0
        self.min_cache = torch.full(
            (self.max_position_embeddings, self.max_position_embeddings),
            torch.finfo(torch.half).min, dtype=torch.half).npu()
        self.in_beta = torch.zeros(config.hidden_size).npu().half()

    def roll_back_layer_append_weight(self, anti_weight_name_list, anti_weight_no_bias_list, layer_index):
        layer_type_list = ['weight', 'bias']
        for anti_weight_name in anti_weight_name_list:
            if anti_weight_name not in anti_weight_no_bias_list:
                for layer_type in layer_type_list:
                    append_item = anti_weight_name.format(layer_index, layer_type)
                    self.weights.append(self.antiOutlier_dict[append_item].half().npu())
            else:
                append_item = anti_weight_name.format(layer_index, 'weight')
                self.weights.append(self.antiOutlier_dict[append_item].half().npu())

    def init_ascend_weight(self):
        quant_weight_dict = np.load(os.path.join(self.weight_path, "quant_weight.npy"), allow_pickle=True).item()
        deq_scale_dict = np.load(os.path.join(self.weight_path, "deq_scale.npy"), allow_pickle=True).item()
        fp_bias_dict = np.load(os.path.join(self.weight_path, "fp_bias_corr.npy"), allow_pickle=True).item()
        self.weights = [self.antiOutlier_dict['model.embed_tokens.weight'].half().npu()]

        layer_type_list = ['weight', 'bias']

        anti_input_layernorm_name = "model.layers.{}.input_layernorm.{}"
        anti_q_proj_name = "model.layers.{}.self_attn.q_proj.{}"
        anti_k_proj_name = "model.layers.{}.self_attn.k_proj.{}"
        anti_v_proj_name = "model.layers.{}.self_attn.v_proj.{}"
        anti_o_proj_name = "model.layers.{}.self_attn.o_proj.{}"
        anti_post_attention_layernorm_name = "model.layers.{}.post_attention_layernorm.{}"
        anti_gate_proj_name = "model.layers.{}.mlp.gate_proj.{}"
        anti_down_proj_name = "model.layers.{}.mlp.down_proj.{}"
        anti_up_proj_name = "model.layers.{}.mlp.up_proj.{}"

        anti_weight_name_list = [anti_input_layernorm_name, anti_q_proj_name, anti_k_proj_name, anti_v_proj_name,
                                 anti_o_proj_name, anti_post_attention_layernorm_name, anti_gate_proj_name,
                                 anti_down_proj_name, anti_up_proj_name]

        anti_weight_no_bias_list = [anti_down_proj_name]

        for layer_index in range(self.num_layers):
            if layer_index in self.float_layers:
                # 回退层
                self.roll_back_layer_append_weight(anti_weight_name_list, anti_weight_no_bias_list, layer_index)
            else:
                # 量化层
                for item in layer_type_list:
                    append_item = anti_input_layernorm_name.format(layer_index, item)
                    self.weights.append(self.antiOutlier_dict[append_item].half().npu())

                q_proj_name = "model.layers.{}.self_attn.q_proj".format(layer_index)
                k_proj_name = "model.layers.{}.self_attn.k_proj".format(layer_index)
                v_proj_name = "model.layers.{}.self_attn.v_proj".format(layer_index)
                o_proj_name = "model.layers.{}.self_attn.o_proj".format(layer_index)
                gate_proj_name = "model.layers.{}.mlp.gate_proj".format(layer_index)
                down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
                up_proj_name = "model.layers.{}.mlp.up_proj".format(layer_index)

                # q
                self.weights.append(quant_weight_dict[q_proj_name].to(torch.int8).npu())
                self.weights.append(padding_descale(deq_scale_dict[q_proj_name]).to(torch.float32).npu())
                self.weights.append(fp_bias_dict[q_proj_name].to(torch.int32).npu())

                # k
                self.weights.append(quant_weight_dict[k_proj_name].to(torch.int8).npu())
                self.weights.append(padding_descale(deq_scale_dict[k_proj_name]).to(torch.float32).npu())
                self.weights.append(fp_bias_dict[k_proj_name].to(torch.int32).npu())

                # v
                self.weights.append(quant_weight_dict[v_proj_name].to(torch.int8).npu())
                self.weights.append(padding_descale(deq_scale_dict[v_proj_name]).to(torch.float32).npu())
                self.weights.append(fp_bias_dict[v_proj_name].to(torch.int32).npu())

                # o
                self.weights.append(quant_weight_dict[o_proj_name].to(torch.int8).npu())
                self.weights.append(padding_descale(deq_scale_dict[o_proj_name]).to(torch.float32).npu())
                self.weights.append(fp_bias_dict[o_proj_name].to(torch.int32).npu())

                # rms norm float
                for item in layer_type_list:
                    append_item = anti_post_attention_layernorm_name.format(layer_index, item)
                    self.weights.append(self.antiOutlier_dict[append_item].half().npu())

                # up
                self.weights.append(quant_weight_dict[up_proj_name].to(torch.int8).npu())
                self.weights.append(padding_descale(deq_scale_dict[up_proj_name]).to(torch.float32).npu())
                self.weights.append(fp_bias_dict[up_proj_name].to(torch.int32).npu())

                # gate
                self.weights.append(quant_weight_dict[gate_proj_name].to(torch.int8).npu())
                self.weights.append(padding_descale(deq_scale_dict[gate_proj_name]).to(torch.float32).npu())
                self.weights.append(fp_bias_dict[gate_proj_name].to(torch.int32).npu())

                # down
                self.weights.append(quant_weight_dict[down_proj_name].to(torch.int8).npu())
                self.weights.append(padding_descale(deq_scale_dict[down_proj_name]).to(torch.float32).npu())
                self.weights.append(fp_bias_dict[down_proj_name].to(torch.int32).npu())

        anti_norm_weight_name = "model.norm.{}"
        for item in layer_type_list:
            append_item = anti_norm_weight_name.format(item)
            self.weights.append(self.antiOutlier_dict[append_item].half().npu())

        self.weights.append(self.lm_head_weight)
        self.acl_fa_operation.set_weight(self.weights)

    def prepare_inputs_for_ascend(self, input_ids, position_ids, attention_mask=None, past_key_values=None):
        self.kv_attention_manager.is_full = not past_key_values
        placeholder = torch.ones(1).npu()
        cos_table, sin_table = self.ascend_rotary_embedding(input_ids, self.kv_attention_manager.token_offset)
        cos_table = cos_table.npu()
        sin_table = sin_table.npu()
        position_ids = position_ids.npu()
        cos_embed = torch.nn.functional.embedding(position_ids, cos_table)
        sin_embed = torch.nn.functional.embedding(position_ids, sin_table)
        inputs = [
                     input_ids,
                     cos_embed.half().npu(),
                     sin_embed.half().npu(),
                     self.kv_attention_manager.get_attention_mask(attention_mask),
                     self.kv_attention_manager.k_cache_input,
                     self.kv_attention_manager.v_cache_input,
                     self.kv_attention_manager.token_offset_tensor,
                     self.kv_attention_manager.seq_len_tensor,
                     self.in_beta,
                     placeholder,
                 ] + self.layer_id_list

        return inputs

    def execute_ascend_operator(self, input_ids, position_ids, attention_mask=None, past_key_values=None):
        acl_inputs = self.prepare_inputs_for_ascend(input_ids, position_ids, attention_mask, past_key_values)
        tmp_param = json.dumps(
            {
                "tokenOffset": self.kv_attention_manager.token_offset_list,
                "seqLen": self.kv_attention_manager.seq_len_list
            })
        acl_model_out = self.acl_fa_operation.execute(acl_inputs, tmp_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(INTERNLM_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        # flash attention init
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.kv_attention_manager = KVAttentionManager(self.config, batch_size, self.rankSize)
            self.kv_attention_manager.min_cache = self.min_cache
            self.attention_mask_max_inc = torch.zeros(
                (self.batch_size, self.max_position_embeddings, self.max_position_embeddings), dtype=torch.half).npu()

        if past_key_values is None:
            # 假设输入batch的长度一样
            self.kv_attention_manager.init_seq_len_and_token_offset(seq_length)
            self.kv_attention_manager.init_attention_mask()
            self.attention_mask_max_inc.zero_()

        if past_key_values is not None:
            # 
            # past_key_values_length = past_key_values[0][0].shape[2]
            # seq_length_with_past = seq_length_with_past + past_key_values_length
            past_key_values_length = self.kv_attention_manager.token_offset
            seq_length_with_past = seq_length_with_past + past_key_values_length
            self.kv_attention_manager.token_offset = self.kv_attention_manager.token_offset + 1
            self.kv_attention_manager.token_offset_tensor += 1

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

        if not past_key_values:  # 使用fa时，在计算首token时会同时计算增量额attention_mask
            self.kv_attention_manager.ori_len_list = attention_mask.sum(dim=-1)
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # add acl model
        if not self.weights:
            self.init_ascend_weight()

        hidden_states = self.execute_ascend_operator(input_ids,
                                                     position_ids,
                                                     attention_mask,
                                                     past_key_values)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = (
            (self.kv_attention_manager.k_cache_input, self.kv_attention_manager.v_cache_input),) if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class InternLMForCausalLM(InternLMPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLMModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # for ascend
        self.lm_head_weight = None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(INTERNLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, InternLMForCausalLM

        >>> model = InternLMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        if self.model.lm_head_weight is None:
            soc_version = torch_npu._C._npu_get_soc_version()
            if soc_version not in [104, 220, 221, 222, 223, 224]:
                self.model.lm_head_weight.data = torch_npu.npu_format_cast(self.lm_head_weight.data, 29)
            self.model.lm_head_weight = self.lm_head.weight.data

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states)
        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = []):
        prompt = ""
        for record in history:
            prompt += f"""<|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        return tokenizer([prompt], return_tensors="pt")

    @torch.no_grad()
    def chat(self,
             tokenizer,
             query: str,
             history: List[Tuple[str, str]] = [],
             streamer: Optional[BaseStreamer] = None,
             max_new_tokens: int = 1024,
             do_sample: bool = True,
             temperature: float = 0.8,
             top_p: float = 0.8,
             **kwargs):
        inputs = self.build_inputs(tokenizer, query, history)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.generate(**inputs,
                                streamer=streamer,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                                **kwargs)
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<eoa>")[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(self,
                    tokenizer,
                    query: str,
                    history: List[Tuple[str, str]] = [],
                    max_new_tokens: int = 1024,
                    do_sample: bool = True,
                    temperature: float = 0.8,
                    top_p: float = 0.8,
                    **kwargs):
        """
        Return a generator in format: (response, history)
        Eg.
        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')])
        ('你好，有什么可以帮助您的吗？', [('你好', '你好，有什么可以帮助您的吗？')])
        """

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ""
                self.received_inputs = False
                self.queue.put((self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                token = self.tokenizer.decode([value[-1]], skip_special_tokens=True)
                if token.strip() != "<eoa>":
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()


@add_start_docstrings(
    """
    The InternLM Model transformer with a sequence classification head on top (linear layer).

    [`InternLMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    INTERNLM_START_DOCSTRING,
)
class InternLMForSequenceClassification(InternLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(INTERNLM_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )