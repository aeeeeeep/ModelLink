import torch
import torch_npu
import numpy as np
import ascendspeed_te_ops
torch.set_printoptions(profile="full")

class UnpadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, seqlen, maxseqlen):
        """unpad forward function
        
        Args:
            input_tensor(tensor): float16, The input matrix, shape（batch,max_seq_len,hidden_size）
            seqlen(list): int32, shape param
            maxseqlen: int32, shape param

        Return:
            output_tensor(tensor): float16, The output matrix, shape（sum(seq_len),hidden_size）
        """
        ctx.maxseqlen = maxseqlen
        ctx.seqlen = seqlen
        qseqlen = torch.tensor(seqlen, device=input_tensor.device, dtype=torch.int32)
        output_tensor = ascendspeed_te_ops.npu_unpad_seqlen(input_tensor, seqlen, maxseqlen)
        return output_tensor

    @staticmethod
    def backward(ctx, input_tensor):
        """unpad backward function
        
        Args:
            input_tensor(tensor): float16, The input matrix, shape（sum(seq_len),hidden_size）
            seq_len(list): int32, shape param
            maxseqlen: int32, shape param

        Return:
            output_tensor(tensor): float16, The output matrix, shape（batch,max_seq_len,hidden_size）
        """
        output_tensor = ascendspeed_te_ops.npu_pad_seqlen(input_tensor.contiguous(), ctx.seqlen, ctx.maxseqlen)
        return output_tensor, None, None

class Unpad(torch.nn.Module):
    def __init__(self, maxseqlen):
        super(Unpad, self).__init__()
        self.maxseqlen = maxseqlen

    def forward(self, input_tensor, seqlen):
        output_tensor = UnpadFunction.apply(input_tensor, seqlen, self.maxseqlen)
        return output_tensor