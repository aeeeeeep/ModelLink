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
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# FP8 amax reduction group.
_AMAX_REDUCTION_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# For DeepSpeed's sequence parallel
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

# This group includes processes for both data and sequence parallelisms.
# We use this group to reduce gradients and shard parameters and optimizer stages for ZeRO.
_SEQUENCE_DATA_PARALLEL_GROUP = None
_SEQUENCE_DATA_PARALLEL_WORLD_SIZE = None
_SEQUENCE_DATA_PARALLEL_RANK = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        sequence_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        use_fp8: bool = False,
        use_distributed_optimizer: bool = False,
) -> None:
    world_size: int = torch.distributed.get_world_size()

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    enable_ds_sequence_parallel = sequence_parallel_size > 1
    if enable_ds_sequence_parallel:
        if world_size % sequence_parallel_size != 0:
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by sequence_parallel_size {sequence_parallel_size})"
            )

    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * sequence_parallel_size)
    sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    num_sequence_data_parallel_groups: int = world_size // sequence_parallel_size // data_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError("pipeline-model-parallel size should be greater than 2 with " "interleaved schedule")
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups

        if sequence_parallel_size > 1:
            tp_or_sp_size = sequence_parallel_size
        else:
            tp_or_sp_size = tensor_model_parallel_size

        for j in range(tp_or_sp_size):
            ranks = range(start_rank + j, end_rank, tp_or_sp_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if use_distributed_optimizer:
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            else:
                group_gloo = None
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

    # Build the sequence data parallel groups.
    global _SEQUENCE_DATA_PARALLEL_GROUP
    all_data_sequence_parallel_group_ranks = []
    if enable_ds_sequence_parallel:
        for i in range(num_sequence_data_parallel_groups):
            ranks = range(i * sequence_data_parallel_size,
                          (i + 1) * sequence_data_parallel_size)
            group = torch.distributed.new_group(ranks)
            all_data_sequence_parallel_group_ranks.append(list(ranks))
            if rank in ranks:
                _SEQUENCE_DATA_PARALLEL_GROUP = group
    else:
        _SEQUENCE_DATA_PARALLEL_GROUP = _DATA_PARALLEL_GROUP

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    num_model_parallel_groups = sequence_data_parallel_size if enable_ds_sequence_parallel else data_parallel_size
    model_parallel_group_ranks = all_data_sequence_parallel_group_ranks if enable_ds_sequence_parallel else all_data_parallel_group_ranks
    for i in range(num_model_parallel_groups):
        ranks = [parallel_group_ranks[i] for parallel_group_ranks in model_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank], ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the FP8 groups.
    global _AMAX_REDUCTION_GROUP
    if use_fp8:
        amax_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_amax_groups: int = world_size // amax_group_size
        for i in range(num_amax_groups):
            start_rank = i * amax_group_size
            end_rank = (i + 1) * amax_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _AMAX_REDUCTION_GROUP = group

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_last_rank():
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
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
    torch.distributed._all_gather_base(gathered, tensor,
                                       group=get_tensor_model_parallel_group())
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
        rank = torch.distributed.get_rank()
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
            aa = get_pipeline_model_parallel_last_rank()
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
