# gen_attention_mask对外接口
输入：
- input：必选输入，数据类型float16, shape（batch,1,max_seq_len,max_seq_len）

输出：
- output_tensor：必选输出，数据类型float16, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）

属性：
- seqLen：必选输入，数据类型int32，vector类型，长度为batch
- headSize：必选输入，数据类型int32

## 案例
```python
import torch
import torch_npu
import ascendspeed_ops

def golden_calc(in_tensors, seqlen, headNum):
    out = []
    for i, s in enumerate(seqlen):
        for _ in range(headNum):
            out.append(in_tensors[i, :, :s, :s].flatten())
    return [torch.hstack(out)]

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

if __name__ == '__main__':
    batch = 2
    maxseqlen = 4
    seqlen = [2, 3]
    headNum = 2
    a = torch.randint(1, 10, (batch, 1, maxseqlen, maxseqlen)).npu().half()
    out = []
    shapeOut = sum(map(lambda x: x**2, seqlen)) * headNum
    for _ in range(shapeOut):
        out.append(0.1)
    result = ascendspeed_ops.npu_genattentionmask(a, seqlen, headNum)
    res = result.cpu().numpy().tolist()
    out_tensors = [result]
    golden_out_tensors = golden_calc(a, seqlen, headNum)
    res_compare = golden_compare(out_tensors, golden_out_tensors)
```
