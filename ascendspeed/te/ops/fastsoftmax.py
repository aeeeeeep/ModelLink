import torch
import torch_npu
import ascendspeed_te_ops


class FastSoftMaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data_input, seq_len, head_num):
        data_output = ascendspeed_te_ops.npu_fastsoftmax(data_input, seq_len, head_num)
        ctx.save_for_backward(data_output)
        ctx.seq_len = seq_len
        ctx.head_num = head_num
        return data_output

    @staticmethod
    def backward(ctx, y_grad):
        y_input, = ctx.saved_tensors
        x_grad = ascendspeed_te_ops.npu_fastsoftmaxgrad(y_input, y_grad, ctx.seq_len, ctx.head_num)
        return x_grad, None, None


class FastSoftMax(torch.nn.Module):
    def __init__(self):
        super(FastSoftMax, self).__init__()

    def forward(self, x: torch.Tensor, seq_len: list, head_num: int):
        return FastSoftMaxFunction.apply(x, seq_len, head_num)

    # def forward(self, x: torch.Tensor, seq_len: list, head_num: int):
    #     attention_scores_tensors_scale = x
    #     attention_probs = x
    #     seq_lengths = seq_len
    #     start = 0
    #     for i, seq_length in enumerate(seq_lengths):
    #         end = start + seq_length * seq_length * head_num
    #         attention_probs[start:end] = torch.softmax(attention_scores_tensors_scale[start:end].view(head_num, seq_length, seq_length ), dim=2).flatten()
    #         start = end
    #     return attention_probs


if __name__ == '__main__':
    def gen_softmax_data(head_num, seq_len):
        x = torch.randn([head_num * seq_len, seq_len]).to(torch.float16)
        w = torch.randn_like(x)
        x.requires_grad = True
        y = torch.softmax(x.to(torch.float32), dim=-1).to(torch.float16)
        y.retain_grad()
        loss = (w * y).sum()
        loss.backward()
        return (x.detach(), w.detach(), y.detach(), x.grad.detach())

    def golden_compare(out_tensors, golden_out_tensors):
        compare_result = [torch.allclose(out.float(), golden.float(), rtol=0.001,
            atol=0.001) for out, golden in zip(out_tensors, golden_out_tensors)]
        return all(compare_result)

    torch.manual_seed(0)
    fast_softmax = FastSoftMax()
    batch_size_imm = 4
    head_num_imm = 8
    for _ in range(5):
        seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
        x_list = []
        y_golden_list = []
        y_grad_list = []
        x_grad_golden_list = []
        w_list = []
        for sl in seq_len:
            x_, w_, y_, x_grad_ = gen_softmax_data(head_num_imm, sl)
            x_list.append(x_.reshape(-1))
            w_list.append(w_.reshape(-1))
            y_golden_list.append(y_.reshape(-1))
            x_grad_golden_list.append(x_grad_.reshape(-1))
        y_golden = torch.cat(y_golden_list)
        x_grad_golden = torch.cat(x_grad_golden_list)

        x = torch.cat(x_list).npu()
        w = torch.cat(w_list).npu()
        x.requires_grad = True
        y = fast_softmax(x, seq_len.tolist(), head_num_imm)
        y.retain_grad()
        loss = (w * y).sum()
        loss.backward()

        out_tensors = [y.detach().cpu(), x.grad.detach().cpu()]
        golden_out_tensors = [y_golden, x_grad_golden]
        success = golden_compare(out_tensors, golden_out_tensors)
        print(success)
        if not success:
            print(f"y = {y.detach().cpu()}")
            print(f"y_golden = {y_golden}")
            print(f"x_grad = {x.grad.detach().cpu()}")
            print(f"x_grad_golden = {x_grad_golden}")
