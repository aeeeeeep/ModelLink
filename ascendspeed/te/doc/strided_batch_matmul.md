# strided_batch_matmul对外接口
输入：
- input1：必选输入，数据类型float16
(1) bmm1、bmm1_grad2、bmm2_grad1： shape（sum(seq_len),hidden_size）
(2) bmm1_grad1、bmm2、bmm2_grad2： shape（headNum*seq_len[0]*seq_len[0]+...+headNum*seq_len[batch-1]*seq_len[batch-1]）
- input2：必选输入，数据类型float16
(1) bmm1、bmm1_grad1、bmm2、bmm2_grad1、bmm2_grad2： shape（sum(seq_len),hidden_size）
(2) bmm1_grad2： shape（headNum*seq_len[0]*seq_len[0]+...+headNum*seq_len[batch-1]*seq_len[batch-1]）

输出：
- output_tensor：必选输出，数据类型float16, shape（headNum*m[0]*n[0]+...+headNum*m[batch-1]*n[batch-1]）

属性：
- transA：必选输入，数据类型int32
- transB：必选输入，数据类型int32
- m：必选输入，数据类型int32，vector类型，长度为batch
- k：必选输入，数据类型int32，vector类型，长度为batch
- n：必选输入，数据类型int32，vector类型，长度为batch
- lda：必选输入，数据类型int32，vector类型，长度为batch
- ldb：必选输入，数据类型int32，vector类型，长度为batch
- ldc：必选输入，数据类型int32，vector类型，长度为batch
- strideA：必选输入，数据类型int32，vector类型，长度为batch
- strideB：必选输入，数据类型int32，vector类型，长度为batch
- strideC：必选输入，数据类型int32，vector类型，长度为batch
- batch：必选输入，数据类型int32
- headNum：必选输入，数据类型int32

## 案例
```python 

import torch
import torch_npu
import ascendspeed_ops
import random

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].flatten(), golden_out_tensors[0].flatten(), rtol=0.001, atol=0.001)

def golden_calc(in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum):
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
    result = ascendspeed_ops.npu_stridedbatchmatmul(A, B, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum)

    in_tensors = [A, B]
    out_tensors = [result]
    golden_out_tensors = golden_calc(in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum )

    res_compare = golden_compare(out_tensors, golden_out_tensors)
    print(">>>>>>>>>>>>>>>>>res:", res_compare)

```