import os
import json
import torch
import torch_npu
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

class Bmm1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, seqlen, head_num):

        hidden_size = a.shape[1]
        batch = len(seqlen)
        ctx.seqlen = seqlen
        ctx.head_num = head_num
        ctx.hidden_size = hidden_size
        ctx.batch = batch
        ctx.save_for_backward(a, b)

        head_size = hidden_size // head_num
        seqlen_squared = [x**2 for x in seqlen]

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
        shapeC = (sum(seqlen_squared) * head_num,)
        c = ascendspeed_te_ops.npu_stridedbatchmatmul(a, b, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num)

        # A = a.view(sum(seqlen), head_num, head_size).permute(1, 0, 2).contiguous()
        # B = b.view(sum(seqlen), head_num, head_size).permute(1, 0, 2).contiguous()
        # c_list = []
        # start = 0
        # for i, seq_length in enumerate(seqlen):
        #     end = start + seq_length
        #     c_ = torch.matmul(A[:, start:end, ...], B[:, start:end, ...].transpose(2, 1))
        #     c_list.append(c_.flatten())
        #     start = end
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

        # bmm1_grad1
        transA = 0
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
        grad_x = ascendspeed_te_ops.npu_stridedbatchmatmul(grad_out, y, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num).view(x.shape[0], x.shape[1])

        # A = grad_out.contiguous()
        # B = y.view(sum(seqlen), head_num, head_size).permute(1, 0, 2).contiguous()
        # grad_x_list = []
        # start = 0
        # start1 = 0
        # for i, seq_length in enumerate(seqlen):
        #     end = start + seq_length
        #     end1 = start1 + seq_length * seq_length * head_num
        #     grad_x_ = torch.matmul(A[start1:end1].view(head_num, seq_length, seq_length), B[:, start:end, ...])
        #     grad_x_list.append(grad_x_.permute(1, 0, 2).contiguous().view(seq_length, hidden_size))
        #     start = end
        #     start1 = end1
        # grad_x = torch.cat(grad_x_list, dim=0).contiguous()
    
        # bmm1_grad2
        transA = 1
        transB = 0
        m = [head_size] * batch
        k = seqlen
        n = seqlen
        lda = [hidden_size] * batch
        ldb = seqlen
        ldc = seqlen
        strideA = [head_size] * batch
        strideB = seqlen_squared
        strideC = [s * head_size for s in seqlen]

        grad_y_tmp = ascendspeed_te_ops.npu_stridedbatchmatmul(x, grad_out, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num)

        grad_y_ops_tmp = torch.empty((sum(seqlen) * hidden_size), device=grad_y_tmp.device, dtype=grad_y_tmp.dtype)
        start = 0
        for i, s in enumerate(seqlen):
            end = start + s * hidden_size
            grad_y_ops_tmp[start :end] = grad_y_tmp[start :end].view(hidden_size, s).transpose(1,0).reshape(hidden_size * s)
            start = end
        grad_y_ops = grad_y_ops_tmp.view(sum(seqlen), hidden_size)

        # A = x.view(sum(seqlen), head_num, head_size).permute(1, 0, 2).contiguous()
        # B = grad_out.contiguous()
        # grad_y_list = []
        # start = 0
        # start1 = 0
        # for i, seq_length in enumerate(seqlen):
        #     end = start + seq_length
        #     end1 = start1 + seq_length * seq_length * head_num
        #     grad_y_ = torch.matmul(A[:, start:end, ...].transpose(1,2), B[start1:end1].view(head_num, seq_length, seq_length))
        #     t = grad_y_.permute(2, 0, 1)
        #     t = t.contiguous().view(seq_length, hidden_size)
        #     grad_y_list.append(grad_y_.permute(2, 0, 1).contiguous().view(seq_length, hidden_size))
        #     start = end
        #     start1 = end1
        # grad_y = torch.cat(grad_y_list, dim=0).contiguous()

        # compare_res = golden_compare(grad_y_ops, grad_y)
        # if not compare_res:
        #     print_rank_0("seqlen: {}".format(seqlen))
        #     print_rank_0("head_num: {}".format(head_num))
        #     print_rank_0("hidden_size: {}".format(hidden_size))
        #     print_rank_0("batch: {}".format(batch))
        #     print_rank_0("head_size: {}".format(head_size))

        #     print_rank_0("grad_y_ops.shape: {}".format(grad_y_ops.shape))
        #     print_rank_0("grad_y_tmp.shape: {}".format(grad_y_tmp.shape))
        #     print_rank_0("grad_y.shape: {}".format(grad_y.shape))
        #     print_rank_0("grad_y_ops: {}".format(grad_y_ops.flatten()[0:10]))
        #     print_rank_0("grad_y_tmp: {}".format(grad_y_tmp.flatten()[0:10]))
        #     print_rank_0("grad_y: {}".format(grad_y.flatten()[0:10]))
        # else:
        #     print_rank_0("compare_res: {}".format(compare_res))
        return grad_x, grad_y_ops, None, None

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors.flatten(), golden_out_tensors.flatten(), rtol=0.001, atol=0.001)

class Bmm1(Module):
    def __init__(self, head_num):
        super(Bmm1, self).__init__()
        self.head_num = head_num

    def forward(self, a, b, seqlen):
        return Bmm1Function.apply(a, b, seqlen, self.head_num)

    # def forward(self, a, b, seqlen):
    #     query_layer = a
    #     key_layer = b
    #     seq_lengths = seqlen

    #     query_layer = query_layer.view(query_layer.size(0), self.head_num, -1).permute(1, 0, 2)
    #     key_layer = key_layer.view(key_layer.size(0), self.head_num, -1).permute(1, 0, 2)
    #     attention_scores_list = []
    #     start = 0
    #     for i, seq_length in enumerate(seq_lengths):
    #         end = start + seq_length
    #         output_size = (query_layer.size(1), seq_length, seq_length)
    #         attention_scores = torch.matmul(query_layer[:, start:end, ...], key_layer[:, start:end, ...].transpose(2, 1))
    #         attention_scores_list.append(attention_scores.flatten().contiguous())
    #         start = end
    #     attention_scores_tensors = torch.cat(attention_scores_list, dim=0).flatten().contiguous()
    #     return attention_scores_tensors



if __name__ == '__main__':
    torch.manual_seed(1)
    # batch = random.randint(1, 32)
    # seqlen = [random.randint(1, 2048) for i in range(batch)]
    # head_num = random.randint(8, 32)
    # head_size = random.randint(64, 256)
    batch = 4
    seqlen = [256, 256, 256, 256]
    head_num = 8
    head_size = 128

    sum_seqlen = sum(seqlen)
    hidden_size = head_size * head_num

    A = torch.randint(1, 5, (sum_seqlen, hidden_size)).npu().half()
    B = torch.randint(1, 5, (sum_seqlen, hidden_size)).npu().half()
    # A = torch.randn(sum_seqlen, hidden_size).npu().half()
    # B = torch.randn(sum_seqlen, hidden_size).npu().half()
    A1 = A.clone()
    B1 = B.clone()
    A.requires_grad = True
    B.requires_grad = True

    A1.requires_grad = True
    B1.requires_grad = True

    # 调用matmul正向算子
    bmm1 = Bmm1(head_num)
    C = bmm1(A, B, seqlen)
    for i in range(1000):
        C += bmm1(A, B, seqlen)

    # 定义目标张量
    target = torch.randn((C.shape[0], ), device=C.device, dtype=C.dtype)

    # 调用matmul反向算子
    C.backward(target)


    query_layer = A1.view(sum_seqlen, head_num, -1).permute(1, 0,  2).contiguous()
    key_layer = B1.view(sum_seqlen, head_num, -1).permute(1, 0, 2).contiguous()

    C1_list = []
    start = 0
    for i, seq_length in enumerate(seqlen):
        end = start + seq_length
        C1_ = torch.matmul(
            query_layer[:, start: end, ...],
            key_layer[:, start: end, ...].transpose(1,2))
        C1_list.append(C1_.flatten())
        start = end
    C1 = torch.cat(C1_list, dim=0).contiguous().flatten()

   # 定义目标张量
    C1.backward(target)

    loss1 = ((C1 - target) ** 2).mean()
    loss1.backward()


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
    # print(torch.nonzero(mask_c_grad))
    # print(C[mask_c], C1[mask_c])
    print(A.grad[mask_a_grad], A1.grad[mask_a_grad])
    print(B.grad[mask_b_grad], B1.grad[mask_b_grad])
    # print(C.grad[mask_c_grad], C1.grad[mask_c_grad])

    print(A.grad, A1.grad)
    print(B.grad, B1.grad)

    print(batch)
    print(seqlen)
    print(head_num)
    print(head_size)
