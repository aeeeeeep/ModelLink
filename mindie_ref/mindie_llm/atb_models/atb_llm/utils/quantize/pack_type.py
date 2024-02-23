# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum


class PackType(Enum):
    ALL_FP = 1
    ALL_INT = 2
    ALL_ANTI = 3
    MIX_FP_INT = 4
    MIX_FP_ANTI = 5
