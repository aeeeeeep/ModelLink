# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn


class W8A16LinearStatic(nn.Module):
    def __init__(self, weight, weight_scale, weight_offset):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer('weight', weight.to(torch.int8))
        self.weight_quant_name = 'per_channel'

        self.register_buffer('weight_scale', weight_scale.to(torch.float16))

        if weight_offset is not None:
            self.register_buffer('weight_offset', (-weight_offset).to(torch.float16))
        else:
            self.weight_offset = None