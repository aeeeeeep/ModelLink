import torch
import torch_npu
import ascendspeed_te_ops

class FastSoftMaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data_input, seq_len, head_num):
        """fast softmax forward function
        
        Args:
            data_input(tensor): float16, The input matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
            seq_len(list): int32, shape param
            head_num: int32, shape param

        Return:
            data_output(tensor): float16, The output matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
        """
        data_output = ascendspeed_te_ops.npu_fast_softmax(data_input, seq_len, head_num)
        ctx.save_for_backward(data_output)
        ctx.seq_len = seq_len
        ctx.head_num = head_num
        return data_output

    @staticmethod
    def backward(ctx, y_grad):
        """fast softmax backward function
        
        Args:
            y_input(tensor): float16, The input matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
            y_grad(tensor): float16, The input matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
            seq_len(list): int32, shape param
            head_num: int32, shape param

        Return:
            x_grad(tensor): float16, The output matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
        """
        y_input, = ctx.saved_tensors
        x_grad = ascendspeed_te_ops.npu_fast_softmax_grad(y_input, y_grad, ctx.seq_len, ctx.head_num)
        return x_grad, None, None

class FastSoftMax(torch.nn.Module):
    def __init__(self):
        super(FastSoftMax, self).__init__()

    def forward(self, x: torch.Tensor, seq_len: list, head_num: int):
        return FastSoftMaxFunction.apply(x, seq_len, head_num)
