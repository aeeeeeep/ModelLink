# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum


class PackType(Enum):
    ALL_FP = 1
    ALL_W8A8 = 2
    ALL_W8A8_ANTI = 3
    MIX_W8A8 = 4
    MIX_W8A8_ANTI = 5
    ALL_W8A16 = 6


class LinearType(Enum):
    INVALID = -1
    FP = 0
    INT = 1
