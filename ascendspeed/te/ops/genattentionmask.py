import os
import json
import torch
import torch_npu
import ascendspeed_te_ops
import time

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

class GenAttentionMask(torch.nn.Module):
    def __init__(self, num_attention_heads_per_partition):
        super(GenAttentionMask, self).__init__()
        self.num_attention_heads_per_partition = num_attention_heads_per_partition
        # batch = 4
        # maxseqlen = 2048
        # self.seqlen = [161, 121, 185, 90]
        # self.num_attention_heads_per_partition = 8
        # print(">>>>>>>>>>>>>>>>>start:")
        # self.input_tensor = torch.randint(0, 2, (batch, 1, maxseqlen, maxseqlen)).npu().bool().half()

    def for_test(self, attention_mask, seq_lengths):
        attention_mask_list = []
        for i, seq_len in enumerate(seq_lengths):
            for j in range(self.num_attention_heads_per_partition):
                attention_mask_list.append(attention_mask[i][..., :seq_len, :seq_len].flatten())
        output_test = torch.cat(attention_mask_list, dim=0)
        return output_test


    def forward(self, attention_mask, seq_lengths):
        input_mask = attention_mask.half() #  self.input_tensor


        headNum = self.num_attention_heads_per_partition
        output_size = sum([s**2 for s in seq_lengths]) * headNum

        # print("GenAttentionMask forward headNum:", headNum)
        # print("GenAttentionMask forward seq_lengths:", seq_lengths)
        # print("GenAttentionMask forward attention_mask shape:", attention_mask.shape)
        # print("GenAttentionMask forward attention_mask:", attention_mask)

        # out_attention_mask = torch.zeros((output_size,), device=attention_mask.device, dtype=input_mask.dtype)
        # stream = torch.npu.current_stream()
        # stream.synchronize()
        out_attention_mask = ascendspeed_te_ops.npu_genattentionmask(input_mask, seq_lengths, headNum)
        # stream = torch.npu.current_stream()
        # stream.synchronize()

        output = out_attention_mask > 0.5

        # print(">>>out_attention_mask: ", out_attention_mask)
        return output

    # def forward(self, attention_mask, seq_lengths):
    #     attention_mask_list = []
    #     for i, seq_len in enumerate(seq_lengths):
    #         for j in range(self.num_attention_heads_per_partition):
    #             attention_mask_list.append(attention_mask[i][..., :seq_len, :seq_len].flatten())
    #     attention_mask = torch.cat(attention_mask_list, dim=0)
    #     return attention_mask


def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors.float(), golden_out_tensors.float(), rtol=0.001, atol=0.001)

# def golden_calc(in_tensors, seqlen, headNum):
#     out = []
#     for i, s in enumerate(seqlen):
#         for _ in range(headNum):
#             out.append(in_tensors[i, :, :s, :s].flatten())
#     print("torch.hstack(out):", torch.hstack(out))
#     return [torch.hstack(out)]

def golden_calc(attention_mask, seq_lengths, headNum):
        attention_mask_list = []
        for i, seq_len in enumerate(seq_lengths):
            for j in range(headNum):
                attention_mask_list.append(attention_mask[i][..., :seq_len, :seq_len].flatten())
        attention_mask = torch.cat(attention_mask_list, dim=0)
        return attention_mask


if __name__ == '__main__':
    for i in range(1):
        batch = 4
        headNum = 8
        seqlen = [161, 121, 185, 90]
        maxseqlen = 2048
        print(">>>>>>>>>>>>>>>>>start:")
        a = torch.randint(0, 2, (batch, 1, maxseqlen, maxseqlen)).npu().bool()

        # print(">>>>>>>>>>>>>>>>>a: ", a )
        gen = GenAttentionMask(headNum)
        result = gen(a, seqlen)
        res = result.cpu().numpy().tolist()
        # print(">>>>>>>>>>>>>>>>>res:", res)
        attention_mask1 = result # [result]
        print(">>>>>>>>>>>>>>>>>res:", result)
        attention_mask = golden_calc(a, seqlen, headNum)
        print(">>>>>>>>>>>>>>>>>attention_mask:", attention_mask)
        # res_compare = golden_compare(out_tensors, golden_out_tensors)
        res_compare = torch.allclose(attention_mask.float(), attention_mask1.float(), rtol=0.001, atol=0.001)
        print("res_compare:", res_compare)

