# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn
from torch.functional import F

from atb_llm.common.log.logging import logger


class FastLinear(nn.Module):
    def __init__(
            self,
            weight,
            bias,
            is_norm=False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        self.is_norm_head = is_norm
        self.first_flag = True

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_norm_head:
            if self.first_flag:
                self.first_flag = False
                self.weight = nn.Parameter(F.normalize(self.weight))
                logger.info(f"do normalize weight for norm head")
            return F.linear(input, self.weight, self.bias)

        return F.linear(input, self.weight, self.bias)
