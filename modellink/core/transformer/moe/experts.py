# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps
import torch.nn.functional as F
from megatron.training import get_args
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu


def groupedmlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args_ = get_args()
        if self.config.gated_linear_unit and args_.use_fused_swiglu:
            if self.config.activation_func != F.silu:
                raise ValueError('When using fused_swiglu, activation function must be silu.')
            self.activation_func = fused_swiglu

    return wrapper
