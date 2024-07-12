
import torch

from megatron.training import get_args
from megatron.core import parallel_state

from mindspeed.core.parallel_state import (get_context_parallel_for_hybrid_ulysses_world_size,
                                             get_context_parallel_for_hybrid_ulysses_rank,
                                             get_context_parallel_for_hybrid_ring_world_size,
                                             get_context_parallel_for_hybrid_ring_rank)


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    args = get_args()
    if args.context_parallel_algo == 'megatron_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_megatron_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_hybrid_cp(pos_emb, seq_dim)
    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_megatron_cp(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    pos_emb = pos_emb.chunk(cp_size, dim=seq_dim)[cp_rank]

    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_hybrid_cp(pos_emb, seq_dim):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()
    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    cp_idx = torch.tensor(
        [r_rank, (2 * r_size - r_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * r_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])

    pos_emb = pos_emb.chunk(u_size, dim=seq_dim)[u_rank]

    return pos_emb
