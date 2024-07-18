# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

from torch import Tensor
import torch_npu
from megatron.training import get_args
from modellink.model.transformer import do_context_parallel, get_attention_mask

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs['config']
        cp_size = config.context_parallel_size
        config.context_parallel_size = 1
        fn(self, *args, **kwargs)
        config.context_parallel_size = cp_size

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
        return do_context_parallel(
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
