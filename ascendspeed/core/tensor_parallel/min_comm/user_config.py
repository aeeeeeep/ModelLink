import importlib
import os
from ascendspeed.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from ascendspeed.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank
)
from ascendspeed.core.tensor_parallel.mappings import (
    _reduce,
    _reduce_scatter_along_first_dim,
    _gather_along_first_dim
)
from .min_comm_cfg import min_comm_config, ModuleType
from .cc_parallel_linears_all_reduce import CCColumnAllReduceFunction, CCRowAllReduceFunction
from .cc_parallel_linears_sequence_parallel import CCColumnSeqParallelFunction, CCRowSeqParallelFunction
from .rewrite_parallel_linears_all_reduce import RewriteColumnAllReduceFunction, RewriteRowAllReduceFunction
from .rewrite_parallel_linears_sequence_parallel import RewriteColumnSeqParallelFunction, RewriteRowSeqParallelFunction


cc_cfgs = {
    'recompute_all_gather': True,
    'matmul_soc_friendly': True,
    'print_tensor_value_open': False,
    'customized_cc': {},
}


def check_config_valid():
    if min_comm_config.sequence_parallel_enabled:
        if min_comm_config.module_type not in [ModuleType.ORIGINAL_SEQ_PARALLEL,
                                               ModuleType.REWRITE_SEQ_PARALLEL,
                                               ModuleType.CC_FOR_SEQ_PARALLEL]:
            raise RuntimeError("In CoC, the config of sequence parallel is not valid")
    else:
        if min_comm_config.module_type not in [ModuleType.ORIGINAL_ALL_REDUCE,
                                               ModuleType.REWRITE_ALL_REDUCE,
                                               ModuleType.CC_FOR_ALL_REDUCE]:
            raise RuntimeError("In CoC, the config of sequence parallel is not valid")


def get_value_from_cfg(attr_name):
    if attr_name not in cc_cfgs.keys():
        raise RuntimeError("Lack attr_name: ", attr_name)
    return cc_cfgs[attr_name]


def initialize_cc_from_cfg(cfg):
    min_comm_config.register_tp_get_functions(get_tensor_model_parallel_group,
                                              get_tensor_model_parallel_world_size,
                                              get_tensor_model_parallel_rank)
    min_comm_config.register_class(ColumnParallelLinear,
                                   RowParallelLinear)
    min_comm_config.register_mappings(_reduce,
                                      _reduce_scatter_along_first_dim,
                                      _gather_along_first_dim)
    min_comm_config.register_sequence_parallel_switch(cfg.sequence_parallel)

    min_comm_config.register_customized_cc(get_value_from_cfg('customized_cc'))
    min_comm_config.register_matmul_soc_friendly_switch(get_value_from_cfg('matmul_soc_friendly'))
    min_comm_config.register_all_gather_recomputation_switch(get_value_from_cfg('recompute_all_gather'))
    min_comm_config.register_print_tensor_value_switch(get_value_from_cfg('print_tensor_value_open'))
    min_comm_config.register_check_fcn(check_config_valid)
    min_comm_config.acquire_module_type(cfg.tensor_model_parallel_size)

    map_type2autograd_class = {
        ModuleType.REWRITE_SEQ_PARALLEL: [RewriteColumnSeqParallelFunction,
                                          RewriteRowSeqParallelFunction],
        ModuleType.REWRITE_ALL_REDUCE: [RewriteColumnAllReduceFunction,
                                        RewriteRowAllReduceFunction],
        ModuleType.CC_FOR_SEQ_PARALLEL: [CCColumnSeqParallelFunction,
                                         CCRowSeqParallelFunction],
        ModuleType.CC_FOR_ALL_REDUCE: [CCColumnAllReduceFunction,
                                       CCRowAllReduceFunction]
    }

    if "ORIGINAL" not in min_comm_config.module_type.name:
        parallel_linear_autograd_class = map_type2autograd_class.get(min_comm_config.module_type)
        if parallel_linear_autograd_class is None:
            raise RuntimeError("Module type is not matched.")
        min_comm_config.replace_forward_functions_by_autograd_class(parallel_linear_autograd_class[0],
                                                                    parallel_linear_autograd_class[1])
