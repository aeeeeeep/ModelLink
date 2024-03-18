import math
from functools import reduce
import operator
from typing import Optional
import torch

"""Model and data parallel groups."""

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None


def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1
) -> None:
    world_size: int = torch.distributed.get_world_size()

    if world_size != (tensor_model_parallel_size * pipeline_model_parallel_size):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    rank = torch.distributed.get_rank()

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    for i in range(0, world_size, tensor_model_parallel_size):
        ranks = range(i, i + tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS

    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def split_tensor_into_1d_equal_chunks(tensor):
    partition_size = torch.numel(tensor) // get_tensor_model_parallel_world_size()
    start_index = partition_size * get_tensor_model_parallel_rank()
    end_index = start_index + partition_size
    data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor):
    numel_gathered = torch.numel(tensor) * get_tensor_model_parallel_world_size()
    gathered = torch.empty(numel_gathered, dtype=tensor.dtype,
                           device=torch.npu.current_device(),
                           requires_grad=False)
    torch.distributed._all_gather_base(gathered, tensor, group=get_tensor_model_parallel_group())
    return gathered


class PipeBaseModule(torch.nn.Module):
    def __init__(self):
        super(PipeBaseModule, self).__init__()
        self.pp_num = get_pipeline_model_parallel_world_size()
        self.pp_rank = get_pipeline_model_parallel_rank()
        self.local_stage_layers_idx = None

    def process_pipe(self, layers):
        num_layers = len(layers)
        stage_layers_num = math.ceil(num_layers / self.pp_num)
        offset = self.pp_rank * stage_layers_num
        end = min(num_layers, offset + stage_layers_num)
        self.local_stage_layers_idx = list(range(offset, end))
        return [layer if layer_id in self.local_stage_layers_idx else None for layer_id, layer in enumerate(layers)]

    def recv(self, input_ids):
        if self.pp_rank == 0:
            h = self.first_stage_process(input_ids)
        else:
            bsz, seqlen = input_ids.shape[0], input_ids.shape[1]
            tensor_shape = (seqlen, bsz, self.dim())
            tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1) // get_tensor_model_parallel_world_size()
            h = torch.empty(
                tensor_chunk_shape, device=self.get_device(), dtype=torch.float16
            )
            torch.distributed.recv(h, src=get_pipeline_model_parallel_prev_rank())
            seq_length, bsz = input_ids.shape[:2]
            tensor_shape = (seq_length, bsz, self.dim())
            h = gather_split_1d_tensor(h).view(tensor_shape)
        return h

    def send(self, h):
        rank = torch.distributed.get_rank()
        if self.pp_rank < self.pp_num - 1:
            bs = h.shape
            h = split_tensor_into_1d_equal_chunks(h)
            torch.distributed.send(h.half(), dst=get_pipeline_model_parallel_next_rank())
            outs = torch.empty(*bs, device=self.get_device(), dtype=torch.float16)
        else:
            outs = self.last_stage_process(h).half()
        if self.pp_num > 1:
            ww = torch.distributed.get_world_size()
            torch.distributed.broadcast(outs, src=ww - 1)
        return outs

    def dim(self):
        raise NotImplementedError("子类必须实现")

    def get_device(self):
        raise NotImplementedError("子类必须实现")

    def get_dtype(self):
        raise NotImplementedError("子类必须实现")

    def first_stage_process(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现")

    def last_stage_process(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现")
