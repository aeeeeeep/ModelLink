import os
from functools import wraps
import torch
import torch_npu
from torch import nn
from megatron.model.rms_norm import RMSNorm
from megatron import get_args


def rms_norm_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args = get_args()
        self.use_fused_rmsnorm = args.use_fused_rmsnorm
        self.add_rmsnorm_offset = args.add_rmsnorm_offset
    return wrapper


def rms_norm_forward(self, x):
    if int(os.getenv('NPU_ASD_ENABLE', '0')):
        from torch_npu.utils import register_asd_hook
        register_asd_hook(x, self.weight)
    if self.use_fused_rmsnorm:
        weight = self.weight if not self.add_rmsnorm_offset else (1 + self.weight)
        return torch_npu.npu_rms_norm(x, weight, epsilon=self.eps)[0]

    output = self._norm(x.float()).type_as(x)
    return output * (self.weight if not self.add_rmsnorm_offset else (1 + self.weight))
