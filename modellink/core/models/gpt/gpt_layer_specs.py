import types
from functools import wraps
from megatron.core.transformer.moe.moe_layer import MoELayer
from mindspeed.core.transformer.custom_layers.transformer_engine import PTNorm
from megatron.training import get_args


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False):
        res = fn(num_experts, moe_grouped_gemm, qk_layernorm)
        args_pos_norm = get_args()
        res.submodules.input_layernorm = PTNorm
        if qk_layernorm:
            res.submodules.self_attention.submodules.q_layernorm = PTNorm
            res.submodules.self_attention.submodules.k_layernorm = PTNorm
        res.submodules.pre_mlp_layernorm = PTNorm
        if args_pos_norm.post_norm:
            res.submodules.post_attn_norm = PTNorm
            res.submodules.post_mlp_layernorm = PTNorm
        return res

    return wrapper


def build_layers_wrapper(fn, column_forward, row_forward):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        for layer in self.layers:
            if isinstance(layer.mlp, MoELayer):
                for local_expert in layer.mlp.experts.local_experts:
                    local_expert.linear_fc1.forward = types.MethodType(column_forward, local_expert.linear_fc1)
                    local_expert.linear_fc2.forward = types.MethodType(row_forward, local_expert.linear_fc2)
    return wrapper
