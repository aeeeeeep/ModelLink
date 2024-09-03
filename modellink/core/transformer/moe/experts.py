# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps
import torch.nn.functional as F
from megatron.training import get_args
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from modellink.core.transformer.mlp import should_recompute_activation


def groupedmlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args_ = get_args()
        if self.config.gated_linear_unit and args_.use_fused_swiglu:
            if self.config.activation_func != F.silu:
                raise ValueError('When using fused_swiglu, activation function must be silu.')
            self.activation_func = fused_swiglu
        self.set_recompute_activation_func = False

    return wrapper


def groupedmlp_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.should_recompute_activation = should_recompute_activation(self)
        if self.should_recompute_activation and not self.set_recompute_activation_func:
            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            self.local_activation_func = self.activation_func

            def recompute_activation_func(*args):
                output = self.activation_checkpoint_manager.checkpoint(self.local_activation_func, False, *args)
                return output
            self.activation_func = recompute_activation_func
            self.set_recompute_activation_func = True
        fc2_output, _ = fn(self, *args, **kwargs)

        if self.should_recompute_activation:
            # discard the activation output and restored by recomputation before backward of fc2.
            self.activation_checkpoint_manager.discard_output()
            if fc2_output.requires_grad:
                fc2_output.register_hook(self.activation_checkpoint_manager.recompute)

        return fc2_output, None
    return wrapper
