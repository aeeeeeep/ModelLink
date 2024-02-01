import torch
import torch_npu
import ascendspeed_te_ops
import numpy as np
from unittest import TestCase
import logging

def golden_calc(data_input, seq_len_list, max_seq_len_imm, hidden_size_imm):
    golden = torch.empty(size=[len(seq_len_list), max_seq_len_imm, hidden_size_imm], dtype=torch.float16)
    start = 0
    for i in range(len(seq_len_list)):
        golden[i][:seq_len_list[i]] = data_input[start:start + seq_len_list[i]]
        golden[i][seq_len_list[i]:] = 0
        start = start + seq_len_list[i]
    return golden


def golden_compare(out_tensor, golden_out_tensor):
    return torch.allclose(out_tensor.float(), golden_out_tensor.float(), rtol=0.001, atol=0.001)


if __name__ == '__main__':
    print(">>>>>>>>>>>>>>>>>tests/ut/test_pad.py start")
    for i in range(10):
        batch = 4
        maxseqlen = 4096
        hidden_size_imm = 4096
        seq_len = torch.randint(low=100, high=300, size=[batch,], dtype=torch.int32)
        seqlen = np.array(seq_len)
        data_input = torch.randn(size=[seq_len.sum(), hidden_size_imm], dtype=torch.float16).npu().half()

        result = ascendspeed_te_ops.npu_pad(data_input, seqlen, maxseqlen)

        res = result.cpu()
        out_tensor = res
        golden_out_tensor = golden_calc(data_input, seqlen, maxseqlen, hidden_size_imm)
        res_compare = golden_compare(out_tensor, golden_out_tensor)
        print("res_compare:", res_compare)