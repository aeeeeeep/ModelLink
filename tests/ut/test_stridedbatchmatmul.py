
import torch
import torch_npu
import ascendspeed_te_ops

from unittest import TestCase
import logging
import random

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].flatten(), golden_out_tensors[0].flatten(), rtol=0.001, atol=0.001)

# def test_add_bmm1(self): #bmm2_grad1
def golden_calc(in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum):
    A = in_tensors[0].flatten()
    B = in_tensors[1].flatten()

    startA = 0
    endA = 0
    startB = 0
    endB = 0
    startC = 0
    endC = 0
    batchStartA = 0
    batchStartB = 0
    batchStartC = 0
    listA = []
    listB = []
    C = torch.zeros(sum([ m[i] * n[i] for i in range(batch)]) * headNum, dtype=torch.float16, device=A.device)

    for i in range(batch):
        batchStartA = endA
        batchStartB = endB
        batchStartC = endC
        for j in range(headNum):
            listA = []
            listB = []
            rowA = m[i] if not transA else k[i]
            colA = k[i] if not transA else m[i]
            for t in range(rowA):
                startA = lda[i] * t + strideA[i] * j + batchStartA
                endA = startA + colA
                listA.append(A[startA:endA])
            rowB = k[i] if not transB else n[i]
            colB = n[i] if not transB else k[i]
            for t in range(rowB):
                startB = ldb[i] * t +  strideB[i] * j + batchStartB
                endB = startB + colB
                listB.append(B[startB:endB])
            matA = torch.stack(listA)
            matB = torch.stack(listB)
            matA = torch.transpose(matA, 0, 1) if transA else matA
            matB = torch.transpose(matB, 0, 1) if transB else matB
            matC = torch.matmul(matA.float(), matB.float()).half()
            for t in range(matC.shape[0]):
                startC = ldc[i] * t + strideC[i] * j + batchStartC
                endC = startC + matC.shape[1]
                C[startC:endC] = matC[t, :]

    return [C]

#bmm1 bmm2_grad1
def golden_calc1(in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum):
    seq_lengths = m
    A = in_tensors[0].view(sum(seq_lengths), headNum, -1).permute(1, 0, 2).contiguous()
    B = in_tensors[1].view(sum(seq_lengths), headNum, -1).permute(1, 0, 2).contiguous()
    attention_scores_list = []
    start = 0
    for i, seq_length in enumerate(seq_lengths):
        end = start + seq_length
        attention_scores = torch.matmul(A[:, start:end, ...], B[:, start:end, ...].transpose(2, 1))
        attention_scores_list.append(attention_scores.flatten())
        start = end
    C = torch.cat(attention_scores_list, dim=0).contiguous()
    return [C]

if __name__ == '__main__':
    print(">>>>>>>>>>>>>>>>>tests/ut/test_stridedbatchmatmul.py start")
    for index in range(5):
        batch = 4
        seqlen = [165, 165, 165, 165]
        head_num = 8
        head_size = 128

        sum_seqlen = sum(seqlen)
        hidden_size = head_size * head_num
        seqlen_squared = [x**2 for x in seqlen]
        shapeC = (head_num * sum(seqlen_squared), )

        transA = 0
        transB = 1
        m = seqlen
        k = [head_size] * batch
        n = seqlen
        lda = [hidden_size] * batch
        ldb = [hidden_size] * batch
        ldc = seqlen
        strideA = [head_size] * batch
        strideB = [head_size] * batch
        strideC = seqlen_squared

        batch = batch
        headNum = head_num

        A = torch.randn(sum_seqlen, hidden_size).npu().half()
        B = torch.randn(sum_seqlen, hidden_size).npu().half()

        shapeC = (head_num * sum(seqlen_squared), )
        result = ascendspeed_te_ops.npu_stridedbatchmatmul(A, B, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum)

        in_tensors = [A, B]
        out_tensors = [result]
        golden_out_tensors = golden_calc1(in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum )

        res_compare = golden_compare(out_tensors, golden_out_tensors)
        print(">>>>>>>>>>>>>>>>>res:", res_compare)
