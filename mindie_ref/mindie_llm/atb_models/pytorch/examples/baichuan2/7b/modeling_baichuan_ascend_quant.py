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
import json
import math
import os
from contextlib import contextmanager
from threading import Thread
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
import torch_npu
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation.utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging, ContextManagers

from .configuration_baichuan import BaichuanConfig
from .generation_utils import build_chat_input, TextIterStreamer


def is_nd():
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version in [104, 220, 221, 222, 223, 224]


IS_ND = is_nd()


def get_rank_and_world_size():
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except:
        rank = 0
        world_size = 1
    return rank, world_size


RANK, WORLD_SIZE = get_rank_and_world_size()


def load_acl_transformer():
    """
    加载acl transformers
    :return:
    """
    acl_transformer_home_path = os.getenv("ATB_SPEED_HOME_PATH", "")
    if not acl_transformer_home_path or not os.path.exists(acl_transformer_home_path):
        raise RuntimeError("env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
    lib_path = os.path.join(acl_transformer_home_path, "lib/libatb_speed_torch.so")
    torch.classes.load_library(lib_path)


load_acl_transformer()

logger = logging.get_logger(__name__)


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


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    if len(mask.size()) == 3:
        bsz, src_len, _ = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, :, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    else:
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
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


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        import platform
        if platform.machine() == "aarch64":
            self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).double().to(device) / dim))
        else:
            self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
        self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32).to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32).to(x.device)
        elif self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


class AscendRotaryEmbedding(RotaryEmbedding):
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
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class MLP(nn.Module):
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


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.config = config
        self.world_size = WORLD_SIZE
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = int(os.getenv("MAX_SEQ_LEN", config.model_max_length))

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.num_heads = self.num_heads // self.world_size
        self.W_pack = nn.Linear(self.hidden_size, 3 * self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

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

        proj = self.W_pack(hidden_states)
        proj = proj.unflatten(-1, (3, self.num_heads * self.head_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

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
            attn_weights = torch.max(attn_weights,
                                     torch.tensor(torch.finfo(attn_weights.dtype).min).to(attn_weights.device))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config)
        self.world_size = WORLD_SIZE
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size // self.world_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

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
        # reduce
        if self.world_size >= 2:
            torch.distributed.all_reduce(
                hidden_states, op=torch.distributed.ReduceOp.SUM)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BaichuanPreTrainedModel(PreTrainedModel):
    config_class = BaichuanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
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
        if isinstance(module, BaichuanModel):
            module.gradient_checkpointing = value


class KVAttentionManager:
    def __init__(self, config: BaichuanConfig, batch_size):
        self.nz_dim = 16
        self.world_size = WORLD_SIZE
        self.is_full = True
        self.batch_size = batch_size
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size // self.world_size
        self.max_seq_len = int(os.getenv("MAX_SEQ_LEN", config.model_max_length))
        self.token_offset = 1
        self.ori_len_list = []
        self.min_cache = None
        if not IS_ND:
            self.k_cache_input = torch.zeros(self.num_layers,
                                             self.batch_size,  # batch
                                             self.hidden_size // self.nz_dim,
                                             self.max_seq_len,
                                             self.nz_dim,
                                             device="npu",
                                             dtype=torch.half)

            self.v_cache_input = torch.zeros(self.num_layers,
                                             self.batch_size,  # batch
                                             self.hidden_size // self.nz_dim,
                                             self.max_seq_len,
                                             self.nz_dim,
                                             device="npu",
                                             dtype=torch.half)
            self.k_cache_input = torch_npu.npu_format_cast(self.k_cache_input, 29)
            torch.npu.empty_cache()
            self.v_cache_input = torch_npu.npu_format_cast(self.v_cache_input, 29)
        else:
            self.k_cache_input = torch.zeros(self.num_layers,
                                             batch_size,  # batch
                                             self.max_seq_len,
                                             self.hidden_size,
                                             device="npu",
                                             dtype=torch.half)
            self.v_cache_input = torch.zeros(self.num_layers,
                                             batch_size,  # batch
                                             self.max_seq_len,
                                             self.hidden_size,
                                             device="npu",
                                             dtype=torch.half)
        torch.npu.empty_cache()

        self.attention_mask_max = torch.zeros(
            (self.batch_size, self.max_seq_len, self.max_seq_len), device="npu", dtype=torch.half)
        self.attention_mask_max_inc = torch.zeros(
            (self.batch_size, self.max_seq_len, self.max_seq_len), device="npu", dtype=torch.half)

    def init_attention_mask(self):
        if IS_ND:
            self.attention_mask_max.zero_()
            self.attention_mask_max_inc.zero_()
        else:
            self.attention_mask_max.zero_()
            self.attention_mask_max_inc = torch.zeros(
            (self.batch_size, self.max_seq_len, self.max_seq_len), device="npu", dtype=torch.half)

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

    def trans_data(self, tensor):
        """
        :param tensor:
        :return:
        """
        return torch_npu.npu_format_cast(tensor.view(
            self.batch_size, self.max_seq_len,
            self.max_seq_len // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)

    def get_attention_mask(self, attention_mask=None):
        if not self.is_full:
            return self.attention_mask_max_inc
        else:
            for i in range(self.batch_size):
                self.attention_mask_max[i][:self.token_offset, :self.token_offset] = attention_mask[i]
                ori_len = self.ori_len_list[i].item()
                # 左padding
                # self.attention_mask_max_inc[i][:, :self.token_offset - ori_len] = self.min_cache[:, :self.token_offset - ori_len]
                # 右padding
                self.attention_mask_max_inc[i][:, ori_len:self.token_offset] = \
                    self.min_cache[:, ori_len:self.token_offset]
            if not IS_ND:
                self.attention_mask_max_inc = self.trans_data(self.attention_mask_max_inc)
                return self.trans_data(self.attention_mask_max)
            else:
                return self.attention_mask_max


class BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.rank = RANK
        self.world_size = WORLD_SIZE
        self.post_init()
        self.model_weights = {}

        self.layer_id_list = [torch.tensor([i], dtype=torch.int32).npu() for i in range(config.num_hidden_layers)]
        self.place_holder = torch.ones(1).npu()

        self.quant_weight_path = '/home/ctl/models/7b_quant_cut'
        self.cut_float_weight = os.path.join('/data/models/baichuan2/7b/Baichuan2-7B-Chat/part_model/', str(self.rank))
        with open(os.path.join(self.cut_float_weight, 'pytorch_model.bin.index.json')) as f:
            pytorch_model_index = json.load(f)
        self.weight_map = pytorch_model_index.get('weight_map')
        weight_files = list(set(self.weight_map.values()))

        for weight_file in weight_files:
            self.model_weights.update(torch.load(os.path.join(self.cut_float_weight, weight_file), map_location='cpu'))
        self.part_quant_weight_path = os.path.join(self.quant_weight_path, str(self.rank))
        self.input_scale_dict = np.load(os.path.join(self.part_quant_weight_path, "input_scale.npy"),
                                        allow_pickle=True).item()
        self.input_offset_dict = np.load(os.path.join(self.part_quant_weight_path, "input_offset.npy"),
                                         allow_pickle=True).item()
        print(self.input_offset_dict.keys())
        self.roll_back_layer = [1, 2, 3, 7, 9, 10, 15, 21, 29, 30, 31]
        # for ascend init
        self.init_ascend_operations(config)

    def init_ascend_operations(self, config: BaichuanConfig):
        w_pack_input_scale = []
        w_pack_input_offset = []
        o_proj_input_scale = []
        o_proj_input_offset = []
        gate_proj_input_scale = []
        gate_proj_input_offset = []
        down_proj_input_scale = []
        down_proj_input_offset = []
        for layer_index in range(config.num_hidden_layers):
            if layer_index in self.roll_back_layer:
                w_pack_input_scale.append(float(0))
                w_pack_input_offset.append(float(0))
                o_proj_input_scale.append(float(0))
                o_proj_input_offset.append(float(0))
                gate_proj_input_scale.append(float(0))
                gate_proj_input_offset.append(float(0))
                down_proj_input_scale.append(float(0))
                down_proj_input_offset.append(float(0))
            else:
                w_pack_name = "model.layers.{}.self_attn.W_pack".format(layer_index)
                o_proj_name = "model.layers.{}.self_attn.o_proj".format(layer_index)
                gate_proj_name = "model.layers.{}.mlp.gate_proj".format(layer_index)
                down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
                w_pack_input_scale.append(float(1 / self.input_scale_dict[w_pack_name]))
                w_pack_input_offset.append(int(self.input_offset_dict[w_pack_name]))
                o_proj_input_scale.append(float(1 / self.input_scale_dict[o_proj_name]))
                o_proj_input_offset.append(int(self.input_offset_dict[o_proj_name]))
                gate_proj_input_scale.append(float(1 / self.input_scale_dict[gate_proj_name]))
                gate_proj_input_offset.append(int(self.input_offset_dict[gate_proj_name]))
                down_proj_input_scale.append(float(1 / self.input_scale_dict[down_proj_name]))
                down_proj_input_offset.append(int(self.input_offset_dict[down_proj_name]))

        self.acl_param = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": config.num_attention_heads // self.world_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "rank": self.rank,
            "rankSize": self.world_size,
            "backend": os.getenv("BACKEND", "hccl"),
            "w_packInputScale": w_pack_input_scale,
            "w_packInputOffset": w_pack_input_offset,
            "o_projInputScale": o_proj_input_scale,
            "o_projInputOffset": o_proj_input_offset,
            "gate_projInputScale": gate_proj_input_scale,
            "gate_projInputOffset": gate_proj_input_offset,
            "down_projInputScale": down_proj_input_scale,
            "down_projInputOffset": down_proj_input_offset,
            "roll_back_layer": self.roll_back_layer
        })
        self.max_position_embeddings = int(os.getenv("MAX_SEQ_LEN", config.max_position_embeddings))
        self.acl_fa_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_7b_flash_attention_quant_model")

        print((self.acl_param))

        self.acl_fa_operation.set_param(self.acl_param)

        self.ascend_rotary_embedding = AscendRotaryEmbedding(
            config.hidden_size // config.num_attention_heads, max_position_embeddings=self.max_position_embeddings)

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.weights = []
        self.lm_head_weight = None
        self.batch_size = 0
        self.kv_attention_manager = None
        self.min_cache = torch.full(
            (self.max_position_embeddings, self.max_position_embeddings),
            torch.finfo(torch.half).min, dtype=torch.half).npu()
        self.in_beta = torch.zeros(config.hidden_size).half().npu()

    def init_ascend_weight(self):

        transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        transdata_operation.set_param(transdata_param)

        quant_weight_dict = np.load(os.path.join(self.quant_weight_path, str(self.rank), "quant_weight.npy"),
                                    allow_pickle=True).item()
        deq_scale_dict = np.load(os.path.join(self.quant_weight_path, str(self.rank), "deq_scale.npy"),
                                 allow_pickle=True).item()
        quant_bias_dict = np.load(os.path.join(self.quant_weight_path, str(self.rank), "bias.npy"),
                                  allow_pickle=True).item()

        self.weights = [self.model_weights.get('model.embed_tokens.weight').to(torch.float16).npu()]

        for layer_index in range(self.num_layers):
            input_layernorm_name = "model.layers.{}.input_layernorm.weight".format(layer_index)
            post_attention_layernorm_name = "model.layers.{}.post_attention_layernorm.weight".format(layer_index)
            self.weights.append(self.model_weights.get(input_layernorm_name).to(torch.float16).npu())
            if layer_index in self.roll_back_layer:
                w_pack_name = "model.layers.{}.self_attn.W_pack.weight".format(layer_index)
                o_proj_name = "model.layers.{}.self_attn.o_proj.weight".format(layer_index)
                up_proj_name = "model.layers.{}.mlp.up_proj.weight".format(layer_index)
                gate_proj_name = "model.layers.{}.mlp.gate_proj.weight".format(layer_index)
                down_proj_name = "model.layers.{}.mlp.down_proj.weight".format(layer_index)

                self.weights.append(
                    torch_npu.npu_format_cast(self.model_weights.get(w_pack_name).to(torch.float16).npu(), 29))
                self.weights.append(
                    torch_npu.npu_format_cast(self.model_weights.get(o_proj_name).to(torch.float16).npu(), 29))
                self.weights.append(self.model_weights.get(post_attention_layernorm_name).to(torch.float16).npu())
                self.weights.append(
                    torch_npu.npu_format_cast(self.model_weights.get(gate_proj_name).to(torch.float16).npu(), 29))
                self.weights.append(
                    torch_npu.npu_format_cast(self.model_weights.get(down_proj_name).to(torch.float16).npu(), 29))
                self.weights.append(
                    torch_npu.npu_format_cast(self.model_weights.get(up_proj_name).to(torch.float16).npu(), 29))

            else:
                w_pack_name = "model.layers.{}.self_attn.W_pack".format(layer_index)
                o_proj_name = "model.layers.{}.self_attn.o_proj".format(layer_index)
                up_proj_name = "model.layers.{}.mlp.up_proj".format(layer_index)
                gate_proj_name = "model.layers.{}.mlp.gate_proj".format(layer_index)
                down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)

                # int8 weight
                self.weights.append(
                    transdata_operation.execute([quant_weight_dict[w_pack_name].to(torch.int8).npu()])[0])
                self.weights.append(deq_scale_dict[w_pack_name].to(torch.int64).npu())
                self.weights.append(quant_bias_dict[w_pack_name].to(torch.int32).npu())

                self.weights.append(
                    transdata_operation.execute([quant_weight_dict[o_proj_name].to(torch.int8).npu()])[0])
                self.weights.append(deq_scale_dict[o_proj_name].to(torch.int64).npu())
                self.weights.append(quant_bias_dict[o_proj_name].to(torch.int32).npu())

                self.weights.append(
                    transdata_operation.execute([quant_weight_dict[up_proj_name].to(torch.int8).npu()])[0])
                self.weights.append(deq_scale_dict[up_proj_name].to(torch.int64).npu())
                self.weights.append(quant_bias_dict[up_proj_name].to(torch.int32).npu())

                self.weights.append(
                    transdata_operation.execute([quant_weight_dict[gate_proj_name].to(torch.int8).npu()])[0])
                self.weights.append(deq_scale_dict[gate_proj_name].to(torch.int64).npu())
                self.weights.append(quant_bias_dict[gate_proj_name].to(torch.int32).npu())

                self.weights.append(
                    transdata_operation.execute([quant_weight_dict[down_proj_name].to(torch.int8).npu()])[0])
                self.weights.append(deq_scale_dict[down_proj_name].to(torch.int64).npu())
                self.weights.append(quant_bias_dict[down_proj_name].to(torch.int32).npu())
                # rms norm float
                self.weights.append(self.model_weights.get(post_attention_layernorm_name).to(torch.float16).npu())

        self.weights.append(self.model_weights.get('model.norm.weight').to(torch.float16).npu())

        self.weights.append(torch_npu.npu_format_cast(self.model_weights['lm_head.weight'].to(torch.float16).npu(), 29))

        print('+' * 80)
        print('init weights ')
        print(len(self.weights))
        self.acl_fa_operation.set_weight(self.weights)
        print('**********init done*******************')
        torch.npu.empty_cache()

    def prepare_inputs_for_ascend(self, input_ids, position_ids, attention_mask=None,
                                  past_key_values=None):
        self.kv_attention_manager.is_full = not past_key_values
        cos_table, sin_table = self.ascend_rotary_embedding(input_ids, self.kv_attention_manager.token_offset)
        cos_embed = torch.nn.functional.embedding(position_ids, cos_table)
        sin_embed = torch.nn.functional.embedding(position_ids, sin_table)

        inputs = [input_ids,
                  cos_embed,
                  sin_embed,
                  self.kv_attention_manager.get_attention_mask(attention_mask),
                  self.kv_attention_manager.k_cache_input,
                  self.kv_attention_manager.v_cache_input,
                  self.kv_attention_manager.token_offset_tensor,
                  self.kv_attention_manager.seq_len_tensor,
                  self.in_beta,
                  self.place_holder
                  ] + self.layer_id_list

        return inputs

    def execute_ascend_operator(self, input_ids, position_ids, attention_mask=None, past_key_values=None):
        acl_inputs = self.prepare_inputs_for_ascend(input_ids, position_ids, attention_mask, past_key_values)
        tmp_param = json.dumps(
            {"tokenOffset": self.kv_attention_manager.token_offset_list,
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
            self.kv_attention_manager = KVAttentionManager(self.config, batch_size)
            self.kv_attention_manager.min_cache = self.min_cache

        if past_key_values is None:
            self.kv_attention_manager.init_attention_mask()
            # 假设输入batch的长度一样
            self.kv_attention_manager.init_seq_len_and_token_offset(seq_length)

        if past_key_values is not None:
            # print("----->  past key value shape", past_key_values[0][0].shape)
            past_key_values_length = self.kv_attention_manager.token_offset
            seq_length_with_past = seq_length_with_past + past_key_values_length
            # NEW
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


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
        elif self.first_flag:
            self.first_flag = False
            self.weight = nn.Parameter(nn.functional.normalize(self.weight))
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


class BaichuanForCausalLM(BaichuanPreTrainedModel):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.model = BaichuanModel(config)

        self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)
        if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
            try:
                from .quantizer import quantize_offline, init_model_weight_int4
            except ImportError:
                raise ImportError(f"Needs QLinear to run quantize.")
            quantize_offline(self, 4)
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

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            use_safetensors: bool = None,
            **kwargs,
    ):
        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=False,
                proxies=None,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder="",
                _from_auto=False,
                _from_pipeline=None,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
            try:
                from .quantizer import init_model_weight_int4
                from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
                from accelerate.utils import CustomDtype
                from accelerate.utils import get_balanced_memory
            except ImportError:
                raise ImportError(f"Needs import model weight init func to run quantize.")
                # Instantiate model.
            init_contexts = [no_init_weights(_enable=True)]
            init_contexts.append(init_empty_weights())
            with ContextManagers(init_contexts):
                model = cls(config)

            model_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
            state_dict = torch.load(model_file, map_location="cpu")
            model.is_quantized = True

            device_map = kwargs.pop("device_map", None)
            torch_dtype = kwargs.pop("torch_dtype", None)

            if device_map is not None:
                kwargs = {"no_split_module_classes": model._no_split_modules}
                target_dtype = CustomDtype.INT4
                max_memory = get_balanced_memory(
                    model,
                    dtype=target_dtype,
                    low_zero=(device_map == "balanced_low_0"),
                    max_memory=None,
                    **kwargs,
                )
                kwargs["max_memory"] = max_memory
                device_map = infer_auto_device_map(model, dtype=target_dtype, **kwargs)

            model = init_model_weight_int4(config, model, state_dict)

            # Set model in evaluation mode to deactivate DropOut modules by default
            model.eval()
            # If it is a model with generation capabilities, attempt to load the generation config
            if model.can_generate():
                try:
                    model.generation_config = GenerationConfig.from_pretrained(
                        pretrained_model_name_or_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=False,
                        proxies=None,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder="",
                        _from_auto=False,
                        _from_pipeline=None,
                        **kwargs,
                    )
                except (OSError, TypeError):
                    logger.info(
                        "Generation config file not found, using a generation config created from the model config."
                    )
                    pass

            if device_map is not None:
                dispatch_model(model, device_map=device_map)

            return model
        return super(BaichuanForCausalLM, cls).from_pretrained(pretrained_model_name_or_path, *model_args,
                                                               config=config, cache_dir=cache_dir,
                                                               ignore_mismatched_sizes=ignore_mismatched_sizes,
                                                               force_download=force_download,
                                                               local_files_only=local_files_only, token=token,
                                                               revision=revision,
                                                               use_safetensors=use_safetensors, **kwargs)

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
        if self.lm_head_weight is None:
            self.lm_head_weight = nn.functional.normalize(self.state_dict()["lm_head.weight"])
            if not IS_ND:
                self.lm_head_weight.data = torch_npu.npu_format_cast(self.lm_head_weight.data, 29)
            self.model.lm_head_weight = self.lm_head_weight

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

    def quantize(self, bits: int):
        try:
            from .quantizer import quantize_online
        except ImportError:
            raise ImportError(f"Needs QLinear to run quantize.")
        return quantize_online(self, bits)

    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig] = None):
        generation_config = generation_config or self.generation_config
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.generate, kwargs=dict(
                inputs=input_ids, streamer=streamer,
                generation_config=generation_config,
            )).start()
            return streamer
        else:
            outputs = self.generate(input_ids, generation_config=generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response
