import torch
import torch_npu
import ascendspeed_te_ops


def maybe_align(x, alignment_in_bytes=16):
    """Assume that x already has last dim divisible by alignment_in_bytes"""
    return x if x.data_ptr() % alignment_in_bytes == 0 else x.clone()


class NPUDropoutAddLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, weight, residual, bias=None, rowscale=None, layerscale=None,
                p=0.0, eps=1e-5, prenorm=False, residual_in_fp32=False,
                is_rms_norm=False, return_dropout_mask=False):
        
        x0 = maybe_align(x0.contiguous(), 16)
        residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
        weight = maybe_align(weight.contiguous(), 16)
        bias = maybe_align(bias.contiguous(), 16) if bias is not None else None
        rowscale = maybe_align(rowscale.contiguous(), 16) if rowscale is not None else None
        colscale = maybe_align(colscale.contiguous(), 16) if colscale is not None else None

        outputs = ascendspeed_te_ops.npu_dropout_add_layer_norm(
            x0, weight, residual, bias, rowscale, layerscale,
            p, eps, prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask)
        norm_result, pre_norm_result, mask_result = outputs

        ctx.p = p
        ctx.eps = eps
        ctx.prenorm = prenorm
        ctx.residual_in_fp32 = residual_in_fp32
        ctx.is_rms_norm = is_rms_norm
        ctx.return_dropout_mask = return_dropout_mask

        return outputs


npu_dropout_add_layer_norm = NPUDropoutAddLayerNorm.apply
