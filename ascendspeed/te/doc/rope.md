# rope对外接口 todo
## 前向接口：
输入：
- input1：必选输入，Q矩阵，数据类型float16, shape（sum(seq_len),hidden_size_q）
    hidden_size_q=head_num*head_size，且是hidden_size_k的整数倍
- input1：必选输入，K矩阵，数据类型float16, shape（sum(seq_len),hidden_size_k）
- input3：必选输入，cos矩阵，数据类型float16, shape（max_seq_len,head_size）
- input4：必选输入，sin矩阵，数据类型float16, shape（max_seq_len,head_size）
- input5：必选输入，tensor格式的seq_len，数据类型int32, shape（batch）

输出：
- output_tensor1：必选输出，ropeQ矩阵，数据类型float16, shape（sum(seq_len),hidden_size_q）
- output_tensor2：必选输出，ropeK矩阵，数据类型float16, shape（sum(seq_len),hidden_size_k）

属性：
- rotaryCoeff：必选输入，数据类型int32
- cosFormat：必选输入，数据类型int32

## 反向接口
输入：
- input1：必选输入，ropeQ_grad矩阵，数据类型float16, shape（sum(seq_len),hidden_size）
- input2：必选输入，ropeQ_grad，数据类型float16, shape（sum(seq_len),hidden_size）
- input3：必选输入，cos矩阵，数据类型float16, shape（max_seq_len,head_size）
- input4：必选输入，sin矩阵，数据类型float16, shape（max_seq_len,head_size）

输出：
- output_tensor1：必选输出，q_grad矩阵，数据类型float16, shape（sum(seq_len),hidden_size）
- output_tensor2：必选输出，k_grad矩阵，数据类型float16, shape（sum(seq_len),hidden_size）

属性：
- qSeqLen：必选输入，数据类型int32，vector类型，长度为batch

## 案例
```python 正向接口案例
import torch
import torch_npu
import ascendspeed_ops
import numpy as np
from unittest import TestCase

def rotate_half(x):
    x0, x1 = x.chunk(2, -1)
    return torch.cat((-x1, x0), dim=x0.ndim - 1)

def golden_compare(out_tensors, golden_out_tensors):
    return (torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
        and torch.allclose(out_tensors[1].float(), golden_out_tensors[1].float(), rtol=0.001, atol=0.001))

def test_ops():
    batch = 4
    rotaryCoeff = 2
    cosFormat = 1
    headDim = 128
    maxseqlen = 2048

    seqlen = np.random.randint(1, maxseqlen, size=batch, dtype=np.int32)
    hiddensizeQ = 1024
    hiddensizeK = 1024
    q = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeQ)).astype(np.float16)
    k = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeK)).astype(np.float16)
    cos = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
    sin = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
    q = torch.tensor(q).npu().half().contiguous()
    k = torch.tensor(k).npu().half().contiguous()
    cos = torch.tensor(cos).npu().half().contiguous()
    sin = torch.tensor(sin).npu().half().contiguous()
    seqlen = torch.tensor(seqlen).npu().int().contiguous()
    q_embed = torch.zeros_like(q,device=q.device).half().contiguous()
    k_embed = torch.zeros_like(q,device=q.device).half().contiguous()
    q_embed, k_embed = ascendspeed_ops.npu_rope(q, k, cos, sin, seqlen, rotaryCoeff, cosFormat)
    out_tensors = [q_embed, k_embed]
    golden_out_tensors = golden_calc(q, k, cos, sin,seqlen)
    res_compare = self.golden_compare(out_tensors, golden_out_tensors)
    print("res_compare:", res_compare)

def golden_calc(q, k, cos, sin,seqlen):
    ntoken = q.shape[0]
    batch = seqlen.shape[0]
    hidden_size = q.shape[1]
    head_size = cos.shape[1]
    head_num = hidden_size // head_size
    q_list = []
    k_list = []
    offset = 0
    for i, _ in enumerate(range(batch)):
        cur_seqlen = seqlen[i]
        next_offset = offset + cur_seqlen
        qlayer = q[offset:next_offset].view(cur_seqlen, head_num, head_size)
        klayer = k[offset:next_offset].view(cur_seqlen, head_num, head_size)
        curcos = cos[:cur_seqlen].unsqueeze(1)
        cursin = sin[:cur_seqlen].unsqueeze(1)
        qlayer = (qlayer * curcos) + (rotate_half(qlayer) * cursin)
        klayer = (klayer * curcos) + (rotate_half(klayer) * cursin)
        outq = qlayer.view(cur_seqlen, hidden_size)
        q_list.append(outq)
        outk = klayer.view(cur_seqlen, hidden_size)
        k_list.append(outk)
        offset = next_offset
    q_sum = torch.cat(tuple(q_list), dim=0)
    k_sum = torch.cat(tuple(k_list), dim=0)
    return [q_sum, k_sum]

if __name__ == '__main__':
    test_ops()
```
```python 反向接口案例
import torch
import torch_npu
import ascendspeed_ops
import numpy as np

def golden_compare(out_tensors, golden_out_tensors):
    return (torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
        and torch.allclose(out_tensors[1].float(), golden_out_tensors[1].float(), rtol=0.001, atol=0.001))

def test_ops():
    batch = 16
    headDim = 128
    maxseqlen=200

    seqlen =np.random.randint(1, maxseqlen, size=batch, dtype=np.int32)
    hiddensizeQ = 2048
    hiddensizeK = 2048
    qembedgrad = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeQ)).astype(np.float16)
    kembedgrad = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeK)).astype(np.float16)
    cos = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
    sin = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
    qembedgrad=torch.tensor(qembedgrad).npu().half()
    kembedgrad=torch.tensor(kembedgrad).npu().half()
    cos=torch.tensor(cos).npu().half()
    sin=torch.tensor(sin).npu().half()
    q_grad, k_grad = ascendspeed_ops.npu_rope_grad(qembedgrad, kembedgrad, cos, sin, seqlen)
    out_tensors = [q_grad, k_grad]
    golden_out_tensors = golden_calc(qembedgrad, kembedgrad, cos, sin,seqlen)
    res_compare = self.golden_compare(out_tensors, golden_out_tensors)
    print("res_compare:", res_compare)

def golden_calc(qembedgrad,kembedgrad, cos, sin,seqlen):
    cos_list=[cos[:x, :] for x in seqlen]
    sin_list=[sin[:x, :] for x in seqlen]
    cos = torch.cat(cos_list, dim=0)
    sin = torch.cat(sin_list, dim=0)
    sin1=sin[:,:64]
    sin2=sin[:,64:]
    rohqgsin=torch.cat((sin2,-sin1),dim=-1)
    
    q_grad=torch.zeros_like(qembedgrad)
    bs=int(qembedgrad.shape[1]/128)
    for i in range(bs):
        q_grad[:,i*128:(i+1)*128] = qembedgrad[:,i*128:(i+1)*128] *(cos+rohqgsin)

    k_grad=torch.zeros_like(kembedgrad)
    for i in range(bs):
        k_grad[:,i*128:(i+1)*128] = kembedgrad[:,i*128:(i+1)*128] *(cos+rohqgsin)
    return [q_grad,k_grad]


if __name__ == '__main__':
    test_ops()
```
