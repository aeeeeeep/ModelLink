# fast_softmax对外接口
## 前向接口：
限制条件：
- 输入矩阵：shape为（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
- 输出矩阵：shape为（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）

输入：
- input：必选输入，数据类型float16, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）

输出：
- output_tensor：必选输出，数据类型float16, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）

属性：
- seqLen：必选输入，数据类型int32，vector类型，长度为batch
- headSize：必选输入，数据类型int32

## 反向接口
输入：
- yInput：必选输入，数据类型float16, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
- yGrad：必选输入，数据类型float16, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）

输出：
- output_tensor：必选输出，数据类型float16, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）

属性：
- seqLen：必选输入，数据类型int32，vector类型，长度为batch
- headSize：必选输入，数据类型int32

## 案例
```python 正向接口案例
import torch
import torch_npu
import ascendspeed_ops

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

def test_fastsoftmax():
    batch_size_imm = 4
    head_num_imm = 8
    seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
    data_input_list = [torch.randn(head_num_imm * seq_len[i] * seq_len[i]).to(
        torch.float16) for i in range(batch_size_imm)]
    data_input = torch.cat(data_input_list).contiguous()
    data_output = ascendspeed_ops.npu_fastsoftmax(data_input.npu(), seq_len.tolist(), head_num_imm)
    # calc golden
    golden_list = [torch.softmax(data_input_list[i].reshape(-1, seq_len[i]).to(torch.float32), dim=-1).to(
        torch.float16).reshape(-1) for i in range(batch_size_imm)]
    data_output = data_output.cpu()
    golden = torch.cat(golden_list)
    out_tensors = [data_output]
    golden_out_tensors = [golden]
    success = golden_compare(out_tensors, golden_out_tensors)
    print("res_compare:", success)
    if not success:
        print(data_output)
        print(golden)

if __name__ == '__main__':
    test_fastsoftmax()

```
```python 反向接口案例
import torch
import torch_npu
import ascendspeed_ops

def gen_softmax_grad(head_num, seq_len):
    x = torch.randn([head_num * seq_len, seq_len]).to(torch.float32)
    x.requires_grad = True
    y = torch.softmax(x.to(torch.float32), dim=-1).to(torch.float32)
    y.retain_grad()
    w = torch.randn_like(x).to(torch.float32)
    loss = (w * y).sum()
    loss.backward()
    return (y.detach().to(torch.float16), y.grad.detach().to(torch.float16), x.grad.detach().to(torch.float16))

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

def test_fastsoftmaxgrad():
    batch_size_imm = 4
    head_num_imm = 8
    seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
    y_input_list = []
    y_grad_list = []
    golden_list = []
    for i in range(seq_len.shape[0]):
        yi, yg, gd = gen_softmax_grad(head_num_imm, seq_len[i])
        y_input_list.append(yi.reshape(-1))
        y_grad_list.append(yg.reshape(-1))
        golden_list.append(gd.reshape(-1))
    y_input = torch.cat(y_input_list)
    y_grad = torch.cat(y_grad_list)
    golden = torch.cat(golden_list)
    x_grad = ascendspeed_ops.npu_fastsoftmaxgrad(y_input.npu(), y_grad.npu(), seq_len.tolist(), head_num_imm)
    out_tensors = [x_grad.cpu()]
    golden_out_tensors = [golden]
    success = self.golden_compare(out_tensors, golden_out_tensors)
    print("res_compare:", success)
    if not success:
        print(x_grad)
        print(golden)

if __name__ == '__main__':
    test_fastsoftmaxgrad()
```
