import operator
from functools import reduce

import torch
from deepspeed.accelerator import get_accelerator
import torch.distributed as dist
from ascendspeed.core import tensor_parallel
from ascendspeed import get_args
from ascendspeed.core.pipeline_parallel.p2p_communication import recv_gather

from ascendspeed.core.pipeline_parallel.fold_adaptor.adaptor_parallel_state import (
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_prev_rank_group,
    get_pipeline_model_parallel_next_rank_group
)


def async_communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next):
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)

    if args.sequence_parallel:
        seq_length = args.seq_length // get_tensor_model_parallel_world_size()
        tensor_shape = (seq_length, args.micro_batch_size, args.hidden_size)

    if args.scatter_gather_tensors_in_pipeline and not args.sequence_parallel:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1) // \
            get_tensor_model_parallel_world_size()
    else:
        tensor_chunk_shape = tensor_shape
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float
    if recv_prev:
        tensor_recv_prev = torch.empty(tensor_chunk_shape,
                                       requires_grad=True,
                                       device=get_accelerator().current_device_name(),
                                       dtype=dtype)
    if recv_next:
        tensor_recv_next = torch.empty(tensor_chunk_shape,
                                       requires_grad=True,
                                       device=get_accelerator().current_device_name(),
                                       dtype=dtype)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    if args.scatter_gather_tensors_in_pipeline and not args.sequence_parallel:
        if tensor_send_next is not None:
            tensor_send_next = tensor_parallel.split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = tensor_parallel.split_tensor_into_1d_equal_chunks(tensor_send_prev)

    ops = []
    if tensor_send_prev is not None:
        torch.distributed.isend(tensor_send_prev,
            get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_prev_rank_group())
    if tensor_recv_prev is not None:
        ops.append(torch.distributed.irecv(tensor_recv_prev,
            get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_prev_rank_group()))
    if tensor_send_next is not None:
        torch.distributed.isend(tensor_send_next,
            get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_next_rank_group())
    if tensor_recv_next is not None:
        ops.append(torch.distributed.irecv(tensor_recv_next,
            get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_next_rank_group()))
    return tensor_recv_prev, tensor_recv_next, ops


def async_communicate_group(tensor_send_next, tensor_send_prev, recv_prev, recv_next):
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)

    if args.sequence_parallel:
        seq_length = args.seq_length // get_tensor_model_parallel_world_size()
        tensor_shape = (seq_length, args.micro_batch_size, args.hidden_size)

    if args.scatter_gather_tensors_in_pipeline and not args.sequence_parallel:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1) // \
            get_tensor_model_parallel_world_size()
    else:
        tensor_chunk_shape = tensor_shape
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float
    if recv_prev:
        tensor_recv_prev = torch.empty(tensor_chunk_shape,
                                       requires_grad=True,
                                       device=get_accelerator().current_device_name(),
                                       dtype=dtype)
    if recv_next:
        tensor_recv_next = torch.empty(tensor_chunk_shape,
                                       requires_grad=True,
                                       device=get_accelerator().current_device_name(),
                                       dtype=dtype)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    if args.scatter_gather_tensors_in_pipeline and not args.sequence_parallel:
        if tensor_send_next is not None:
            tensor_send_next = tensor_parallel.split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = tensor_parallel.split_tensor_into_1d_equal_chunks(tensor_send_prev)

    ops = []
    recv_prev_op = None
    send_next_op = None
    recv_next_op = None
    send_prev_op = None
    if get_pipeline_model_parallel_rank() % 2 == 1:
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.irecv(tensor_recv_prev,
                                                   get_pipeline_model_parallel_prev_rank())
        if tensor_send_next is not None:
            send_next_op = torch.distributed.isend(tensor_send_next,
                                                   get_pipeline_model_parallel_next_rank())
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.irecv(tensor_recv_next,
                                                   get_pipeline_model_parallel_next_rank())
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.isend(tensor_send_prev,
                                                   get_pipeline_model_parallel_prev_rank())
    else:
        if tensor_send_next is not None:
            send_next_op = torch.distributed.isend(tensor_send_next,
                                                   get_pipeline_model_parallel_next_rank())
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.irecv(tensor_recv_prev,
                                                   get_pipeline_model_parallel_prev_rank())
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.isend(tensor_send_prev,
                                                   get_pipeline_model_parallel_prev_rank())
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.irecv(tensor_recv_next,
                                                   get_pipeline_model_parallel_next_rank())
    for op in [recv_prev_op, recv_next_op, send_prev_op, send_next_op]:
        if op is not None:
            ops.append(op)

    return tensor_recv_prev, tensor_recv_next, ops

