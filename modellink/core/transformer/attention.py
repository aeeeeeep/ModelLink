# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from functools import wraps
from typing import Union
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer import build_module, TransformerConfig

@dataclass
class MLASelfAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    linear_qb: Union[ModuleSpec, type] = None
    linear_kvb: Union[ModuleSpec, type] = None


def attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)

        args = get_args()
        if args.context_parallel_size > 1 and args.context_parallel_algo in ['ulysses_cp_algo', 'hybrid_cp_algo']:
            ulysses_group = mpu.get_context_parallel_group()
            if args.context_parallel_algo == 'hybrid_cp_algo':
                ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
            self.core_attention = UlyssesContextAttention(self.core_attention, ulysses_group)

        self.use_flash_attn = args.use_flash_attn
        self.multi_head_latent_attention = args.multi_head_latent_attention
        if self.multi_head_latent_attention:
            self.qk_rope_head_dim = args.qk_rope_head_dim
            self.qk_nope_head_dim = args.qk_nope_head_dim
            self.q_lora_rank = args.q_lora_rank
            self.kv_lora_rank = args.kv_lora_rank
            self.v_head_dim = args.v_head_dim

    return wrapper


def self_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self,
                config: TransformerConfig,
                submodules: SelfAttentionSubmodules,
                layer_number: int,
                attn_mask_type=AttnMaskType.padding, ):

        fn(self, config, submodules, layer_number, attn_mask_type)

        if self.multi_head_latent_attention:
            query_projection_size = self.config.num_attention_heads * self.v_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.linear_qkv = build_module(
                submodules.linear_qkv,
                self.config.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv',
            )
            if submodules.q_layernorm is not None:
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    hidden_size=self.q_lora_rank,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.q_layernorm = None

            self.linear_qb = build_module(
                submodules.linear_qb,
                self.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qb',
            )

            if submodules.k_layernorm is not None:
                self.k_layernorm = build_module(
                    submodules.k_layernorm,
                    hidden_size=self.kv_lora_rank,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.k_layernorm = None

            self.linear_kvb = build_module(
                submodules.linear_kvb,
                self.kv_lora_rank,
                self.config.num_attention_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='kvb',
            )

            self.linear_proj = build_module(
                submodules.linear_proj,
                query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='proj',
            )

    return wrapper


def attention_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
):
    """
    Do patch for repeating KV so that GQA+Ulysses is better supported.
    """
    # hidden_states: [sq, b, h]

    # For self attention we just duplicate the rotary_pos_emb if it isn't already
    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        rotary_pos_emb = (rotary_pos_emb,) * 2

    if self.multi_head_latent_attention:
        attn_mask_type = AttnMaskType.no_mask
        q_len, bsz, _ = hidden_states.shape
        mixed_x_layer, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, hn]
        q_a, compressed_kv, k_pe = torch.split(
            mixed_x_layer,
            [
                self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim,
            ],
            dim=-1)

        q, _ = self.linear_qb(self.q_layernorm(q_a))
        q = q.view(q_len, bsz, self.config.num_attention_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        k_pe = k_pe.view(q_len, bsz, 1, self.qk_rope_head_dim)
        kv, _ = self.linear_kvb(self.k_layernorm(compressed_kv))
        kv = kv.view(q_len, bsz, self.config.num_attention_heads, self.qk_nope_head_dim +
                     self.v_head_dim)
        k_nope, value = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            b, h, s, d = q_pe.shape
            q_pe = q_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
            b, h, s, d = k_pe.shape
            k_pe = k_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            q_pe = apply_rotary_pos_emb(q_pe, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            k_pe = apply_rotary_pos_emb(k_pe, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        # query = torch.cat([q_nope, q_pe], dim=-1)
        query = k_pe.new_empty(q_len, bsz, self.config.num_attention_heads, self.q_head_dim)
        query[:, :, :, : self.qk_nope_head_dim] = q_nope
        query[:, :, :, self.qk_nope_head_dim:] = q_pe

        key = k_pe.new_empty(q_len, bsz, self.config.num_attention_heads, self.q_head_dim)
        key[:, :, :, : self.qk_nope_head_dim] = k_nope
        key[:, :, :, self.qk_nope_head_dim:] = k_pe

        if self.use_flash_attn and self.q_head_dim != self.v_head_dim:
            value = F.pad(value, [0, self.q_head_dim - self.v_head_dim])
    else:
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
            )
            key = apply_rotary_pos_emb(
                key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
            )

    # Do repeat KV to support GQA+Ulysses
    args = get_args()
    should_kv_repeat_before_uly = args.context_parallel_size > 1 and \
                           args.context_parallel_algo in ['ulysses_cp_algo', 'hybrid_cp_algo'] and \
                           args.kv_head_repeat_before_uly_alltoall
    heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
    if should_kv_repeat_before_uly and heads_per_gqa_group > 1:
        key = key.repeat_interleave(heads_per_gqa_group, dim=2)
        value = value.repeat_interleave(heads_per_gqa_group, dim=2)

    # ==================================
    # core attention computation
    # ==================================

    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )
    else:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )

    if packed_seq_params is not None:
        # reshape to same output shape as unpacked case
        # (t, np, hn) -> (t, b=1, h=np*hn)
        # t is the pack size = sum (sq_i)
        # note that batch is a dummy dimension in the packed case
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

    if self.multi_head_latent_attention and self.use_flash_attn and self.q_head_dim != self.v_head_dim:
        core_attn_out = core_attn_out.view(q_len, bsz, self.config.num_attention_heads, self.q_head_dim)
        core_attn_out = core_attn_out[:, :, :, : self.v_head_dim]
        core_attn_out = core_attn_out.reshape(q_len, bsz, self.config.num_attention_heads * self.v_head_dim)

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.linear_proj(core_attn_out)

    return output, bias
