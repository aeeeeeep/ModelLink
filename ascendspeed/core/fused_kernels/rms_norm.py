import torch_npu

def rms_norm_forward(self, x):
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]