import torch
import torch_npu
import ascendspeed_te_ops

from unittest import TestCase
import logging

def golden_calc(in_tensors, seqlen, headNum):
    out = []
    for i, s in enumerate(seqlen):
        for _ in range(headNum):
            out.append(in_tensors[i, :, :s, :s].flatten())
    return [torch.hstack(out)]


def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)


if __name__ == '__main__':
    print(">>>>>>>>>>>>>>>>>tests/ut/test_genattentionmask.py start")
    for i in range(10):
        batch = 2
        maxseqlen = 4
        seqlen = [2, 3]
        headNum = 2
        a = torch.randint(1, 10, (batch, 1, maxseqlen, maxseqlen)).npu().half()
        out = []
        shapeOut = sum(map(lambda x: x**2, seqlen)) * headNum
        for _ in range(shapeOut):
            out.append(0.1)

        result = ascendspeed_te_ops.npu_genattentionmask(a, seqlen, headNum)
        res = result.cpu().numpy().tolist()
        out_tensors = [result]
        golden_out_tensors = golden_calc(a, seqlen, headNum)
        res_compare = golden_compare(out_tensors, golden_out_tensors)
        print("res_compare:", res_compare)