# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses


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

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.linear_proj(core_attn_out)

    return output, bias
