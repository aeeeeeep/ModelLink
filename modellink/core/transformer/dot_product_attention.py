# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

import torch
from torch import Tensor
import torch_npu
from megatron.training import get_args
from megatron.core import mpu, parallel_state, tensor_parallel
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks)
from modellink.model.transformer import get_attention_mask
from modellink.core.models.common.embeddings.rotary_pos_embedding import yarn_get_mscale
from modellink.utils import get_actual_seq_len

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def do_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask, dropout_p=0.):
    args = get_args()
    in_hybrid_mode = get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None
    if in_hybrid_mode:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()
    else:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()

    cp_para = dict()

    cp_para['causal'] = args.cp_attention_mask_type == 'causal'
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
        if args.use_cp_send_recv_overlap else None

    output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p)
    return output


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs['config']
        cp_size = config.context_parallel_size
        config.context_parallel_size = 1
        fn(self, *args, **kwargs)
        config.context_parallel_size = cp_size

        args = get_args()
        self.attn_logit_softcapping = args.attn_logit_softcapping
        if args.query_pre_attn_scalar:
            self.norm_factor = args.query_pre_attn_scalar ** 0.5
            self.scale_mask_softmax.scale = 1.0
            self.softmax_scale = 1.0 / self.norm_factor

        if args.multi_head_latent_attention:
            self.scale_mask_softmax.scale = True
            self.hidden_size_per_partition = args.num_attention_heads * args.v_head_dim
            self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
            self.softmax_scale = self.q_head_dim ** (-0.5)

            if args.rope_scaling_type is not None:
                mscale_all_dim = args.rope_scaling_mscale_all_dim if args.rope_scaling_mscale_all_dim else 0
                scaling_factor = args.rope_scaling_factor

                if mscale_all_dim:
                    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                    self.softmax_scale = self.softmax_scale * mscale * mscale

            self.norm_factor = 1.0 / self.softmax_scale

    return wrapper


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params):
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        if get_args().use_flash_attn:
            return flash_attention_forward(self, query, key, value, attention_mask, attn_mask_type,
                                                 packed_seq_params)

        # [b, np, sq, sk]
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        if self.attn_logit_softcapping is not None:
            matmul_result = matmul_result / self.attn_logit_softcapping
            matmul_result = torch.tanh(matmul_result)
            matmul_result = matmul_result * self.attn_logit_softcapping

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context

    return wrapper


def flash_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
):
    if packed_seq_params is not None:
        raise AssertionError("packed_seq_params should be None.")

    args = get_args()

    seq_length, _, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    actual_seq_len = None
    if args.reset_attention_mask or args.reset_position_ids:
        query, key, value = [rearrange(x, 's b h d -> (s b) h d') for x in [query, key, value]]
        args.shape_order = "TND"
        actual_seq_len = get_actual_seq_len()
    else:
        query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
        args.shape_order = "SBH"

    if self.hidden_size_per_attention_head == 0:
        raise AssertionError("self.hidden_size_per_attention_head should not be ZERO.")
    scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) \
        if self.scale_mask_softmax.scale is None else self.softmax_scale
    if attention_mask is None or args.reset_attention_mask or args.reset_position_ids:
        attention_mask = get_attention_mask()

    if args.context_parallel_size > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
        return do_ring_context_parallel(
            query, key, value, head_num=n_head, softmax_scale=scale, attn_mask=attention_mask)
    else:
        use_sliding_windows = args.sliding_window is not None and seq_length > args.sliding_window

        if use_sliding_windows:
            args.pre_tockens = args.sliding_window

        if hasattr(args, 'use_kv_cache') and args.use_kv_cache:
            query, key, value = [rearrange(x, 's b h -> b s h') for x in [query, key, value]]
            if query.shape[1] == 1 and query.shape[1] != key.shape[1]:
                output = torch_npu.npu_incre_flash_attention(
                    query, key, value,
                    num_heads=n_head, 
                    input_layout="BSH",
                    padding_mask=None,
                    scale_value=scale
                )
            else:
                output = torch_npu.npu_prompt_flash_attention(
                    query, key, value,
                    num_heads=n_head, 
                    input_layout="BSH",
                    sparse_mode=args.sparse_mode,
                    padding_mask=None,
                    atten_mask=attention_mask,
                    scale_value=scale,
                    pre_tokens=args.pre_tockens,
                    next_tokens=args.next_tockens
                )
            output = output.transpose(0, 1)
        else:
            output = torch_npu.npu_fusion_attention(
                query, key, value, n_head, 'SBH',
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pre_tockens=args.pre_tockens,
                next_tockens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
            )[0]

        if args.reset_attention_mask or args.reset_position_ids:
            output = rearrange(output, '(s b) h d -> s b (h d)', s=seq_length)

        return output
