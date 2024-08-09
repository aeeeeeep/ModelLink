# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

from torch import Tensor
import torch_npu
from megatron.training import get_args
from megatron.core import mpu
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks)
from modellink.model.transformer import get_attention_mask
from modellink.core.models.common.embeddings.rotary_pos_embedding import yarn_get_mscale

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
        if args.multi_head_latent_attention:
            self.hidden_size_per_partition = args.num_attention_heads * args.v_head_dim
            self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
            self.softmax_scale = self.q_head_dim ** (-0.5)
            
            if args.rope_scaling_type is not None:
                mscale_all_dim = args.rope_scaling_mscale_all_dim if args.rope_scaling_mscale_all_dim else 0
                scaling_factor = args.rope_scaling_factor
                
                if mscale_all_dim:
                    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                    self.softmax_scale = self.softmax_scale * mscale * mscale

    return wrapper


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params):
        if get_args().use_flash_attn:
            return dot_product_attention_forward(self, query, key, value, attention_mask, attn_mask_type,
                                                 packed_seq_params)
        return fn(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)

    return wrapper


def dot_product_attention_forward(
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

    query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]

    if self.hidden_size_per_attention_head == 0:
        raise AssertionError("self.hidden_size_per_attention_head should not be ZERO.")
    scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) \
        if self.scale_mask_softmax.scale is None else self.softmax_scale

    if attention_mask is None:
        attention_mask = get_attention_mask()

    if args.context_parallel_size > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
        return do_ring_context_parallel(
            query, key, value, head_num=n_head, softmax_scale=scale, attn_mask=attention_mask)
    else:
        use_sliding_windows = args.sliding_window is not None and seq_length > args.sliding_window

        if use_sliding_windows:
            args.pre_tockens = args.sliding_window

        return torch_npu.npu_fusion_attention(
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
