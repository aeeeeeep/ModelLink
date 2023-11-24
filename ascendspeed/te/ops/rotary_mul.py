import torch
import torch_npu
import ascendspeed_te_ops


class RotaryMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r1, r2):
        
        ctx.x = x
        ctx.key = key
        ctx.r1 = r1
        ctx.r2 = r2

        outputs = ascendspeed_te_ops.npu_rotary_mul(x, r1, r2)

        ctx.saved_for_backward(x, r1, r2)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        x, r1, r2 = ctx.saved_tensors
        results = ascendspeed_te_ops.npu_rotary_mul_backward(x, r1, r2, grad_outputs)
        return results


rotary_mul = RotaryMul.apply
