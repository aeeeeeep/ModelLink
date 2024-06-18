# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps

import torch
import torch_npu
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half, get_pos_emb_on_this_cp_rank


def RotaryEmbedding_forward(self, max_seq_len: int, offset: int = 0):
    """Forward pass of RoPE embedding.

    Args:
        max_seq_len (int): Maximum size of sequence
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        Tensor: Embeddings after applying RoPE.
    """
    seq = (
        torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        + offset
    )

    if self.seq_len_interpolation_factor is not None:
        seq *= 1 / self.seq_len_interpolation_factor

    freqs = torch.outer(seq, self.inv_freq)
    # first part even vector components, second part odd vector components,
    #  2 * dim in dimension size

    args = get_args()
    if args.use_partial_rope:
        emb = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        if freqs.dtype in (torch.float16, torch.bfloat16, torch.int8):
            emb = emb.bfloat16() if dtype == torch.bfloat16 else emb.half()
    else:
        emb = torch.cat((freqs, freqs), dim=-1)
    # emb [seq_length, .., dim]
    emb = emb[:, None, None, :]
    if parallel_state.get_context_parallel_world_size() > 1:
        # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
        emb = get_pos_emb_on_this_cp_rank(emb, 0)
    return emb


def apply_rotary_pos_emb(t, freqs, rotary_interleaved = False):
    args = get_args()

    # use partial rope in ChatGLM3
    if args.use_partial_rope:
        sq, b, np, hn = t.size(0), t.size(1), t.size(2), t.size(3)
        rot_dim = freqs.shape[-2] * 2
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        freqs = freqs[:sq].to(t.dtype)
        xshaped = t.reshape(sq, -1, np, rot_dim // 2, 2)
        freqs = freqs.view(sq, -1, 1, xshaped.size(3), 2)
        x_shape1, x_shape2 = torch.chunk(xshaped, 2, dim=-1)
        freqs1, freqs2 = torch.chunk(freqs, 2, dim=-1)
        t = torch.stack(
            [
                x_shape1 * freqs1 - x_shape2 * freqs2,
                x_shape2 * freqs1 + x_shape1 * freqs2,
            ],
            -1,
        )
        t = t.flatten(3)
        return torch.cat((t, t_pass), dim=-1)

    if args.use_fused_rotary_pos_emb:
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return torch_npu.npu_rotary_mul(t, cos, sin).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)
    
