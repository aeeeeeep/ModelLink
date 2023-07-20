import torch
import torch_npu
import ext_ops
from torch.autograd import Function
from torch.nn import Module

class _FlashAttention(Function):
    @staticmethod
    def forward(ctx, query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, 
                pre_tockens, next_tockens, gen_mask_parallel, sync):
        flash_func = flashAtten.forward
        attention_score = flash_func(ctx, query, key, value, head_num, input_layout, pse, padding_mask, 
                                  atten_mask, scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)

        return attention_score

    @staticmethod
    def backward(ctx, grad_outputs):
        back_func = flashAtten.backward
        results = back_func(ctx, grad_outputs)
        return results

class FlashAttention(Module):
    def __init__(self):
        super(FlashAttention, self).__init__()
        self.atten = _FlashAttention.apply

    def forward(self, query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None, 
                scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, 
                gen_mask_parallel=True, sync=False):
        return self.atten(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, 
                          scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)