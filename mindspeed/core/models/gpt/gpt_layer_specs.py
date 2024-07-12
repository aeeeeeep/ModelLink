from functools import wraps
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.custom_layers.transformer_engine import TENorm


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False):
        res = fn(num_experts, moe_grouped_gemm, qk_layernorm)
        res.submodules.input_layernorm = TENorm
        if qk_layernorm:
            res.submodules.self_attention.submodules.q_layernorm = TENorm
            res.submodules.self_attention.submodules.k_layernorm = TENorm
        res.submodules.pre_mlp_layernorm = TENorm
        return res

    return wrapper


def get_mlp_module_spec_wrapper(fn, column_forward, row_forward):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if res.module == MoELayer:
            res.submodules.linear_fc1.forward = column_forward
            res.submodules.linear_fc2.forward = row_forward
        return res
    return wrapper
