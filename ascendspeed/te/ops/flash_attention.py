import torch
import torch_npu
import ascendspeed_te_ops


class AscendFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, atten_mask=None, alibi_mask=None,
                scale=1.0, head_num=1, input_layout="BNSD", keep_prob=1.0, pre_tokens=2147483647, next_tokens=1,
                gen_mask_parallel=True, sync=False):
        
        ctx.scale = scale
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        ctx.keep_prob = keep_prob
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens
        ctx.gen_mask_parallel = gen_mask_parallel
        ctx.sync = sync

        outputs = ascendspeed_te_ops.ascend_flash_attention(query, key, value, atten_mask, alibi_mask,
                                                            scale, head_num, input_layout,
                                                            keep_prob, pre_tokens, next_tokens,
                                                            gen_mask_parallel, sync)

        attention_score, softmax_log_max_sum, seed, offset, numels = outputs
        ctx.seed = seed
        ctx.offset = offset
        ctx.numels = numels
        ctx.save_for_backward(
            query, key, value, softmax_log_max_sum, attention_score, atten_mask, alibi_mask
        )

        return attention_score, softmax_log_max_sum

    @staticmethod
    def backward(ctx, grad_output_atten, grad_output_soft):
        query, key, value, softmax_log_max_sum, attention_score, atten_mask, alibi_mask = ctx.saved_tensors
        query_grad, key_grad, value_grad = ascendspeed_te_ops.ascend_flash_attention_grad(
            grad_output_atten, query, key, value, softmax_log_max_sum, attention_score, atten_mask, alibi_mask,
            ctx.scale, ctx.head_num, ctx.input_layout, ctx.keep_prob, ctx.pre_tokens, ctx.next_tokens,
            ctx.seed, ctx.offset, ctx.numels, ctx.gen_mask_parallel, ctx.sync)

        return query_grad, key_grad, value_grad, None, None, None, None, None, None, None, None, None

ascend_flash_attention = AscendFlashAttention.apply
