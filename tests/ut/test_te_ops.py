import torch
import torch_npu
import ascendspeed_te_ops

from unittest import TestCase

class AscendTeOpTest(TestCase):
    def test_add(self):
        a = torch.tensor([1,2]).npu().float()
        b = torch.tensor([2,1]).npu().float()
        result = ascendspeed_te_ops.npu_add(a, b)
        golden = [3., 3.]
        self.assertAlmostEqual(result.cpu().numpy().tolist(), golden, 3)