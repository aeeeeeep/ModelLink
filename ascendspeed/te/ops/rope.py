import os
import json
import torch
import torch_npu
import numpy as np
import ascendspeed_te_ops
torch.set_printoptions(profile="full")

def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RopeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, seqlen):
        rotaryCoeff=2
        cosFormat=1
        ctx.cos = cos
        ctx.sin = sin
        ctx.seqlen=seqlen
        qseqlen = torch.tensor(seqlen, device=q.device, dtype=torch.int32)
        out_tensors = ascendspeed_te_ops.npu_rope(q, k, cos, sin, qseqlen, rotaryCoeff, cosFormat)
        return out_tensors[0],out_tensors[1]

    @staticmethod
    def backward(ctx, q_embed_grad,k_embed_grad):
        out_tensors = ascendspeed_te_ops.npu_rope_grad(q_embed_grad, k_embed_grad,
            ctx.cos, ctx.sin, ctx.seqlen)
        return out_tensors[0], out_tensors[1], None, None, None

class Rope(torch.nn.Module):
    def __init__(self):
        super(Rope, self).__init__()

    def forward(self, q, k, cos, sin, seqlen, offset=0):
        # 处理输入数据，调用Ropefunction
        q_embed, k_embed = RopeFunc.apply(q, k, cos, sin, seqlen)
        return q_embed,k_embed

    # def forward(self, q, k, cos, sin, seqlen, offset=0):
    #     query_layer = q
    #     key_layer = k
    #     seq_lengths = seqlen
    #     query_layer = query_layer.contiguous().view(query_layer.size(0), -1, cos.shape[-1])
    #     key_layer = key_layer.contiguous().view(key_layer.size(0), -1, cos.shape[-1])
    #     query_layer = query_layer.permute(1, 0, 2).contiguous()
    #     key_layer = key_layer.permute(1, 0, 2).contiguous()

    #     cos_list = [cos[None, :x, :] for x in seq_lengths]
    #     sin_list = [sin[None, :x, :] for x in seq_lengths]
    #     cos = torch.cat(cos_list, dim=1)
    #     sin = torch.cat(sin_list, dim=1)
    #     # 32,x,128 #x,4096 reshape.transpose
    #     # 1,x,128
    #     q_embed = (query_layer * cos) + (_rotate_half(query_layer) * sin)
    #     k_embed = (key_layer * cos) + (_rotate_half(key_layer) * sin)
    #     q_embed = q_embed.permute(1, 0, 2).contiguous().view(-1, q.shape[-1])
    #     k_embed = k_embed.permute(1, 0, 2).contiguous().view(-1, k.shape[-1])
    #     return q_embed, k_embed

