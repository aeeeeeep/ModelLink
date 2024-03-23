# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn


class W8A8SparseCompressedLinear(nn.Module):
    def __init__(self, weight, deq_scale, input_scale, quant_bias=None, input_offset=None, index=None):
        super().__init__()
        self.register_buffer('weight', weight.to(torch.int8))

        self.register_buffer('input_scale', input_scale.to(torch.float16))

        self.register_buffer('input_offset', input_offset.to(torch.int8))

        self.register_buffer('deq_scale', deq_scale)

        self.register_buffer('quant_bias', quant_bias)

        self.register_buffer('index', index)
