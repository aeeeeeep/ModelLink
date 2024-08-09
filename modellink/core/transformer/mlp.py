# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.

from functools import wraps
import torch

from megatron.training import get_args
from megatron.core import parallel_state
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.model.transformer import should_recompute_activation


def should_recompute_activation(self):
    args = get_args()
    if not args.recompute_activation_function or self.layer_number is None:
        return False

    activation_recompute_layers = args.recompute_activation_function_num_layers
    vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
    vpp_size = args.virtual_pipeline_model_parallel_size
    pp_size = args.transformer_pipeline_model_parallel_size

    if vpp_size is not None:
        layer_per_chunk = args.num_layers_per_virtual_pipeline_stage
    elif pp_size is not None:
        layer_per_chunk = args.num_layers // pp_size
    else:
        layer_per_chunk = args.num_layers

    if vpp_rank is None or not args.enable_recompute_layers_per_pp_rank:
        vpp_rank = 0
    if vpp_size is None or not args.enable_recompute_layers_per_pp_rank:
        vpp_size = 1
    recompute_priority = ((self.layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    full_recompute_layers = args.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full re-computation when both full re-computation and activation re-computation are enabled
            return False
        elif activation_recompute_layers is None:
            # Do activation function re-computation
            return True
        elif recompute_priority < full_recompute_layers + activation_recompute_layers:
            # Do activation function re-computation
            return True
        else:
            # No recomputation
            return False

    if activation_recompute_layers is None:
        # Do activation function re-computation
        return True
    else:
        return recompute_priority < activation_recompute_layers


def core_mlp_forward_wrapper(fn):
    """
    For mcore mlp re-computation.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        is_recompute_activation = should_recompute_activation(self)

        def activation_function(*function_args):
            intermediate, bias = function_args
            if bias is not None:
                intermediate = intermediate + bias
            if self.config.gated_linear_unit:
                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate = glu(intermediate)
            else:
                intermediate = self.activation_func(intermediate)

            return intermediate

        if not is_recompute_activation:
            output, output_bias = fn(self, *args, **kwargs)
        else:
            hidden_states = args[0]
            intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            intermediate_parallel = self.activation_checkpoint_manager.checkpoint(activation_function,
                                                                                  False,
                                                                                  intermediate_parallel,
                                                                                  bias_parallel)
            # [s, b, h]
            output, output_bias = self.linear_fc2(intermediate_parallel)

            # discard the output of the activation function,
            # which will be restored by re-computation during backward.
            self.activation_checkpoint_manager.discard_output()

            # when backward to output of dense_4h_to_h,
            # recompute and restore the output of activation function.
            if output.requires_grad:
                output.register_hook(self.activation_checkpoint_manager.recompute)
        return output, output_bias

    return wrapper
