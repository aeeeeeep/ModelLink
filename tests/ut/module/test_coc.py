import os
import sys
import unittest
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from wrapt_timeout_decorator import timeout

m = 1024
k = 4096
n = 2048
world_size = 8
tp_group = None
input_tensor_dtype = torch.float32
weight = torch.randn(k, n, dtype=input_tensor_dtype)


class UtCfg:
    def __init__(self, sequence_parallel=True, tp_size=8):
        self.sequence_parallel = sequence_parallel
        self.tensor_model_parallel_size = tp_size


ut_cfg = UtCfg(sequence_parallel=True, tp_size=world_size)


class NotAlmostEqualError(Exception):
    def __init__(self, tensor_a, tensor_b, max_diff, error_info):
        super().__init__()
        self.max_diff = max_diff
        self._error_info = error_info
        self._tensor_a = tensor_a
        self._tensor_b = tensor_b

    def __str__(self):
        if self._error_info is None:
            return f"{self._tensor_a} is not almost equal to {self._tensor_b}, with max_diff {self.max_diff}."
        else:
            return self._error_info.format(self._tensor_a, self._tensor_b)


def shuffle_as_cc_reduce_scatter(input_, world_size_value, parallel_num):
    per = input_.shape[0] // parallel_num // world_size_value
    input_shape = list(input_.shape)
    reshape_tensor = torch.reshape(input_, [parallel_num, world_size_value, per] + input_shape[1:])
    return torch.reshape(reshape_tensor.transpose(0, 1), tuple(input_shape))


def shuffle_as_cc_all_gather(input_, world_size_value, parallel_num):
    per = input_.shape[0] // parallel_num // world_size_value
    input_shape = list(input_.shape)
    reshape_tensor = torch.reshape(input_, [world_size_value, parallel_num, per] + input_shape[1:])
    return torch.reshape(reshape_tensor.transpose(0, 1), tuple(input_shape))


def check_almost_equal(tensor_a, tensor_b, error_bound, error_info=None):
    tensor_diff = torch.abs(torch.sub(tensor_a, tensor_b))
    max_diff = torch.max(tensor_diff)
    if max_diff <= error_bound:
        return
    raise NotAlmostEqualError(tensor_a, tensor_b, max_diff, error_info)


def print_tensor_value(name, value, device_id, split_num):
    if torch.cuda.current_device() == device_id:
        per = value.shape[0] // split_num
        slices = []
        for i in range(split_num):
            v = torch.flatten(value[i * per: (i + 1) * per])
            slices.append(v[:5])
        print(f"Print Tensor: {name}, shape={value.shape}, value=\n{torch.cat(tuple(slices)).view(split_num, -1)}", 
              flush=True)


def get_tp_rank():
    return torch.distributed.get_rank(group=tp_group)


def ut_reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    torch.distributed.all_reduce(input_, group=tp_group)
    return input_


def ut_gather_along_first_dim(input_):
    """Gather tensors and concatenate along the first dimension."""
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(), group=tp_group)
    return output


def ut_reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    dim_size = list(input_.size())
    if dim_size[0] % world_size != 0:
        raise RuntimeError("First dimension of the tensor should be divisible by tensor parallel size")
    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=tp_group)
    return output


def ut_split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the corresponding slice."""
    # Split along first dimension.
    dim_size = input_.size()[0]
    if dim_size % world_size != 0:
        raise RuntimeError("First dimension of the tensor should be divisible by tensor parallel size")
    local_dim_size = dim_size // world_size
    rank = get_tp_rank()
    dim_offset = rank * local_dim_size
    output = input_[dim_offset:dim_offset + local_dim_size].contiguous()
    return output


def initialize(rank, world_size_value, initialize_cc_from_cfg):
    global tp_group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size_value)
    ranks = range(world_size_value)
    tp_group = dist.new_group(ranks)
    initialize_cc_from_cfg(ut_cfg)


def compute(input_tensor, output_tensor=None):
    if output_tensor is None:
        output_tensor = torch.matmul(input_tensor, weight.to(torch.cuda.current_device()))
    else:
        torch.matmul(input_tensor, weight.to(torch.cuda.current_device()), out=output_tensor)
    return output_tensor


def get_original_output_parallel(input_parallel, communicate_fcn, compute_first=True):
    if compute_first:
        compute_output_parallel = compute(input_parallel)
        original_output_parallel = communicate_fcn(compute_output_parallel)
    else:
        communicate_output_parallel = communicate_fcn(input_parallel)
        original_output_parallel = compute(communicate_output_parallel)
    return original_output_parallel


def get_cc_output_parallel(input_parallel, comm_type, compute_first, CCParallel, CommunicationType):
    if comm_type == CommunicationType.REDUCE_SCATTER:
        input_parallel = shuffle_as_cc_all_gather(input_parallel, world_size, 2)
    cc_parallel_class = CCParallel(input_parallel, comm_type=comm_type, compute_fcn=compute,
                                   compute_first=compute_first,
                                   weight_shape_list=[k, n],
                                   parallel_num=2)
    cc_parallel_class.group = tp_group
    cc_parallel_class.tp_world_size = world_size
    cc_output_parallel = cc_parallel_class.run()
    if comm_type == CommunicationType.ALL_GATHER:
        cc_output_parallel = shuffle_as_cc_reduce_scatter(cc_output_parallel, world_size, 2)
    return cc_output_parallel


def generate_tensors(customized_m):
    input_tensor_total = torch.randn(customized_m * world_size, k, dtype=input_tensor_dtype) / 10
    input_parallel = ut_split_along_first_dim(input_tensor_total).to(torch.cuda.current_device())
    return input_parallel


def global_run(rank, world_size_value, customized_m, comm_name, compute_first):
    from ascendspeed.mpu.min_comm.min_comm_cfg import min_comm_config
    from ascendspeed.mpu.min_comm.user_config import initialize_cc_from_cfg
    from ascendspeed.mpu.min_comm.cc_utils import CCParallel, CommunicationType
    get_CCL_type = {
        "ALL_GATHER": CommunicationType.ALL_GATHER,
        "REDUCE_SCATTER": CommunicationType.REDUCE_SCATTER,
        "ALL_REDUCE": CommunicationType.ALL_REDUCE
    }
    get_CCL_fcn = {
        CommunicationType.ALL_GATHER.value: ut_gather_along_first_dim,
        CommunicationType.REDUCE_SCATTER.value: ut_reduce_scatter_along_first_dim,
        CommunicationType.ALL_REDUCE.value: ut_reduce
    }
    comm_type = get_CCL_type.get(comm_name)
    initialize(rank, world_size_value, initialize_cc_from_cfg)
    input_parallel = generate_tensors(customized_m)
    global min_comm_config

    def get_tp_group():
        return tp_group
    
    def get_world_size():
        return world_size_value
    
    min_comm_config.tp_group_fcn = get_tp_group
    min_comm_config.tp_world_size_fcn = get_world_size
    min_comm_config.register_mappings(ut_reduce, ut_reduce_scatter_along_first_dim, ut_gather_along_first_dim)
    cc_output_parallel = get_cc_output_parallel(input_parallel, comm_type, compute_first, CCParallel, 
                                                CommunicationType)
    CCL_fcn = get_CCL_fcn.get(comm_type.value)
    original_output_parallel = get_original_output_parallel(input_parallel, CCL_fcn, compute_first)
    check_almost_equal(original_output_parallel, cc_output_parallel, error_bound=1e-3)


class TestCCParallelClass(unittest.TestCase):
    @timeout(1200)
    def test_reduce_scatter_compute_first(self):
        mp.spawn(global_run, nprocs=world_size, args=(world_size, m * world_size, "REDUCE_SCATTER", True))

    @timeout(1200)
    def test_reduce_scatter_communicate_first(self):
        mp.spawn(global_run, nprocs=world_size, args=(world_size, m * world_size, "REDUCE_SCATTER", False))

    @timeout(1200)
    def test_all_gather_compute_first(self):
        mp.spawn(global_run, nprocs=world_size, args=(world_size, m, "ALL_GATHER", True))

    @timeout(1200)
    def test_all_gather_communicate_first(self):
        mp.spawn(global_run, nprocs=world_size, args=(world_size, m, "ALL_GATHER", False))

    @timeout(1200)
    def test_all_reduce_compute_first(self):
        mp.spawn(global_run, nprocs=world_size, args=(world_size, m * world_size, "ALL_REDUCE", True))

    @timeout(1200)
    def test_all_reduce_communicate_first(self):
        mp.spawn(global_run, nprocs=world_size, args=(world_size, m * world_size, "ALL_REDUCE", False))
