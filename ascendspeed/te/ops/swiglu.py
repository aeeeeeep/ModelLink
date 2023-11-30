import torch
import ascendspeed_te_ops
class NpuSwiGlu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        outputs = ascendspeed_te_ops.npu_swiglu(x, dim)
        ctx.save_for_backward(x)
        ctx.dim = dim
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        x = ctx.saved_tensors[0]
        results = ascendspeed_te_ops.npu_swiglu_grad(grad_outputs, x, ctx.dim)
        return results, None

swiglu = NpuSwiGlu.apply
