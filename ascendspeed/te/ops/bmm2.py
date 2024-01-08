import os
import json
import torch
import torch_npu
from torch.autograd import Function
from torch.nn import Module
import random
import numpy as np
import ascendspeed_te_ops

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

class Bmm2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, seqlen, head_num):

        hidden_size = b.shape[1]
        batch = len(seqlen)
        ctx.seqlen = seqlen
        ctx.head_num = head_num
        ctx.hidden_size = hidden_size
        ctx.batch = batch
        ctx.save_for_backward(a, b)

        head_size = hidden_size // head_num
        seqlen_squared = [x**2 for x in seqlen]

        transA = 0
        transB = 0
        m = seqlen
        k = seqlen
        n =  [head_size] * batch
        lda = seqlen
        ldb = [hidden_size] * batch
        ldc =  [hidden_size] * batch
        strideA = seqlen_squared
        strideB =  [head_size] * batch
        strideC =  [head_size] * batch
        c = ascendspeed_te_ops.npu_stridedbatchmatmul(a, b, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num).view(sum(seqlen), hidden_size)
        # stream = torch.npu.current_stream()
        # stream.synchronize()

        # A = a
        # B = b.view(sum(seqlen), head_num, -1).permute(1, 0, 2).contiguous()
        # c_list = []
        # start = 0
        # start1 = 0
        # for i, seq_length in enumerate(seqlen):
        #     end = start + seq_length
        #     end1 = start1 + seq_length * seq_length * head_num
        #     c_ = torch.matmul(A[start1:end1].view(head_num, seq_length, seq_length), B[:, start:end, ...])
        #     c_list.append(c_.permute(1, 0, 2).contiguous().view(seq_length, hidden_size))
        #     start = end
        #     start1 = end1
        # c = torch.cat(c_list, dim=0).contiguous()

        return c

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        seqlen = ctx.seqlen
        head_num = ctx.head_num
        hidden_size = ctx.hidden_size
        batch = ctx.batch

        head_size = hidden_size // head_num
        seqlen_squared = [x**2 for x in seqlen]

        # bmm2_grad1
        transA = 0
        transB = 1
        m = seqlen
        k = [head_size] * batch
        n =  seqlen
        lda =  [hidden_size] * batch
        ldb = [hidden_size] * batch
        ldc =  seqlen
        strideA = [head_size] * batch
        strideB =  [head_size] * batch
        strideC =  seqlen_squared
        grad_x = ascendspeed_te_ops.npu_stridedbatchmatmul(grad_out, y, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num)

        # A = grad_out.view(sum(seqlen), head_num, -1).permute(1, 0, 2).contiguous()
        # B = y.view(sum(seqlen), head_num, -1).permute(1, 0, 2).contiguous()
        # attention_scores_list = []
        # start = 0
        # for i, seq_length in enumerate(seqlen):
        #     end = start + seq_length
        #     attention_scores = torch.matmul(A[:, start:end, ...], B[:, start:end, ...].transpose(2, 1))
        #     attention_scores_list.append(attention_scores.flatten())
        #     start = end
        # grad_x = torch.cat(attention_scores_list, dim=0).contiguous()

        # bmm2_grad2
        transA = 1
        transB = 0
        m = seqlen
        k = seqlen
        n = [head_size] * batch
        lda = seqlen
        ldb = [hidden_size] * batch
        ldc = [hidden_size] * batch
        strideA = seqlen_squared
        strideB = [head_size] * batch
        strideC = [head_size] * batch
        grad_y = ascendspeed_te_ops.npu_stridedbatchmatmul(x, grad_out, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num).view(sum(seqlen), hidden_size)

        # A = x
        # B = grad_out.view(sum(seqlen), head_num, -1).permute(1, 0, 2).contiguous()
        # grad_y_list = []
        # start = 0
        # start1 = 0
        # for i, seq_length in enumerate(seqlen):
        #     end = start + seq_length
        #     end1 = start1 + seq_length * seq_length * head_num
        #     grad_y_ = torch.matmul(A[start1:end1].view(head_num, seq_length, seq_length).transpose(2, 1), B[:, start:end, ...])
        #     grad_y_list.append(grad_y_.permute(1, 0, 2).contiguous().view(seq_length, hidden_size))
        #     start = end
        #     start1 = end1
        # grad_y = torch.cat(grad_y_list, dim=0).contiguous()
        return grad_x, grad_y, None, None


class Bmm2(Module):
    def __init__(self, head_num):
        super(Bmm2, self).__init__()
        self.head_num = head_num

    def forward(self, a, b, seqlen):
        return Bmm2Function.apply(a, b, seqlen, self.head_num)

    # def forward(self, a, b, seqlen):
    #     attention_probs = a
    #     value_layer = b
    #     seq_lengths = seqlen
    #     value_layer = value_layer.view(value_layer.size(0), self.head_num, -1).permute(1, 0, 2).contiguous()
    #     context_layer_list = []
    #     value_layer_start = 0
    #     attention_probs_start = 0
    #     for i, seq_length in enumerate(seq_lengths):
    #         value_layer_end = value_layer_start + seq_length
    #         attention_probs_end = attention_probs_start + value_layer.shape[0] * seq_length * seq_length
    #         context_layer = torch.matmul(
    #             attention_probs[attention_probs_start:attention_probs_end].view(
    #                 value_layer.shape[0], seq_length, seq_length),
    #             value_layer[:, value_layer_start:value_layer_end, ...])

    #         nh, sq, hd = context_layer.shape
    #         context_layer = context_layer.permute(1, 0, 2).contiguous()
    #         context_layer = context_layer.view(sq, nh * hd)
    #         context_layer_list.append(context_layer)
    #         value_layer_start = value_layer_end
    #         attention_probs_start = attention_probs_end
    #     context_layer = torch.cat(context_layer_list, dim=0).contiguous()
    #     return context_layer

if __name__ == '__main__':
    torch.manual_seed(1)
    batch = random.randint(1, 32)
    seqlen = [random.randint(1, 2048) for i in range(batch)]
    head_num = random.randint(8, 32)
    head_size = random.randint(64, 256)
    # batch = random.randint(1, 5)
    # seqlen = [random.randint(1, 64) for i in range(batch)]
    # head_num = random.randint(2, 8)
    # head_size = random.randint(2, 64)
    batch = 31
    seqlen = [793, 1779, 1942, 1810, 1775, 852, 1682, 2018, 1635, 124, 658, 469, 822, 1094, 1781, 562, 397, 1409, 2031, 1005, 1575, 1761, 1186, 1539, 460, 1137, 1776, 491, 1144, 1446, 823]
    head_num = 24
    head_size = 249


    sum_seqlen = sum(seqlen)
    hidden_size = head_size * head_num
    seqlen_squared = [x**2 for x in seqlen]
    shapeA = (head_num * sum(seqlen_squared), )
    A = torch.randn(shapeA).npu().half()

    B = torch.randn(sum_seqlen, hidden_size).npu().half()

    A1 = A.clone()
    B1 = B.clone()

    A.requires_grad = True
    B.requires_grad = True
    A1.requires_grad = True
    B1.requires_grad = True

    bmm2 = Bmm2(head_num)
    # 调用matmul正向算子
    C = bmm2(A, B, seqlen)
    # for i in range(1000):
    #     C += bmm2(A, B, seqlen)
    target = torch.randn((C.shape[0], C.shape[1]), device=C.device, dtype=C.dtype)

    C.backward(target)

    value_layer = B1.view(sum_seqlen, head_num, -1).permute(1, 0, 2).contiguous()
    C1_list = []
    value_layer_start = 0
    A1_start = 0
    for i, seq_length in enumerate(seqlen):
        value_layer_end = value_layer_start + seq_length
        A1_end = A1_start + head_num * seq_length * seq_length
        # =========================
        # Context layer. [np, sq, hn]
        # =========================
        C1_ = torch.matmul(
            A1[A1_start:A1_end].view(
                head_num, seq_length, seq_length),
            value_layer[:, value_layer_start:value_layer_end, ...])
        nh, sq, hd = C1_.shape
        C1_ = C1_.permute(1, 0, 2).contiguous()
        C1_ = C1_.view(sq, nh * hd)
        C1_list.append(C1_)
        value_layer_start = value_layer_end
        A1_start = A1_end
    C1 = torch.cat(C1_list, dim=0).contiguous()
    # 定义目标张量
    C1.backward(target)

    mask_c =  C != C1
    mask_a_grad = A.grad != A1.grad
    mask_b_grad = B.grad != B1.grad
    mask_c_grad = C.grad != C1.grad
    print(torch.equal(C, C1))
    print(torch.equal(A.grad, A1.grad))
    print(torch.equal(B.grad, B1.grad))
    print(torch.allclose(C, C1, rtol=0.001, atol=0.001))
    print(torch.allclose(A.grad, A1.grad, rtol=0.001, atol=0.001))
    print(torch.allclose(B.grad, B1.grad, rtol=0.001, atol=0.001))
    print(torch.nonzero(mask_c))
    print(torch.nonzero(mask_a_grad))
    print(torch.nonzero(mask_b_grad))
    print(A.grad[mask_a_grad], A1.grad[mask_a_grad])
    print(B.grad[mask_b_grad], B1.grad[mask_b_grad])

    print(batch)
    print(seqlen)
    print(head_num)
    print(head_size)