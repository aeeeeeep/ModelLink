from functools import wraps
import torch
import megatron
from megatron import get_args
from megatron.model import transformer
from megatron.core import tensor_parallel
from megatron.arguments import core_transformer_config_from_args


def Internlm_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        config = args[0]
        query_projection_size = config.kv_channels * config.num_attention_heads
        _args = get_args()
        if _args.group_query_attention:
            kv_projection_size = _args.kv_channels * _args.num_query_groups
        else:
            kv_projection_size = _args.kv_channels * _args.num_attention_heads
        bias = getattr(config, "column_parallel_linear_bias", _args.add_bias_linear)
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=False)
        #   ≈‰internlmƒ£–Õ
        bias = getattr(config, "row_parallel_linear_bias", _args.add_bias_linear)
        skip_bias_add = getattr(config, "row_parallel_linear_skip_bias_add", True)
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add)
    return wrapper

def apply_Internlm_patch():
    megatron.model.transformer.ParallelAttention.__init__ = Internlm_wrapper(megatron.model.transformer.ParallelAttention.__init__)
    