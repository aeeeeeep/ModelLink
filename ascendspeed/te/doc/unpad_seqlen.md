# unpad_seqlen对外接口
输入：
- input：必选输入，数据类型float16, shape（batch,max_seq_len,hidden_size）

输出：
- output_tensor：必选输出，数据类型float16, shape（sum(seq_len),hidden_size）

属性：
- seqLen：必选输入，数据类型int32，vector类型，长度为batch
- maxSeqLen：必选输入，数据类型int32

## 案例
```python 
import torch
import torch_npu
import ascendspeed_ops
import numpy as np

def golden_calc(data_input, seq_len_list, max_seq_len_imm, hidden_size_imm):
    golden = torch.empty(size=[sum(seq_len_list), hidden_size_imm], dtype=torch.float16)
    start = 0
    for i in range(len(seq_len_list)):
        golden[start:start + seq_len_list[i]] = data_input[i][:seq_len_list[i]]
        start = start + seq_len_list[i]
    return golden

def golden_compare(out_tensor, golden_out_tensor):
    return torch.allclose(out_tensor.float(), golden_out_tensor.float(), rtol=0.001, atol=0.001)

if __name__ == '__main__':
    batch = 4
    maxseqlen = 4096
    hidden_size_imm = 4096
    seq_len = torch.randint(low=100, high=300, size=[batch,], dtype=torch.int32)
    seqlen = np.array(seq_len)
    data_input = torch.randn(size=[batch, maxseqlen, hidden_size_imm], dtype=torch.float16).npu().half()

    result = ascendspeed_ops.npu_unpad(data_input, seqlen, maxseqlen)

    res = result.cpu()
    out_tensor = res
    golden_out_tensor = golden_calc(data_input, seqlen, maxseqlen, hidden_size_imm)
    res_compare = golden_compare(out_tensor, golden_out_tensor)
    print("res_compare:", res_compare)
```
