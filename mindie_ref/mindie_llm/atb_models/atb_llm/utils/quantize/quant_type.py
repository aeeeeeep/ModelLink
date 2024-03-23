# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum


class QuantType(str, Enum):
    FLOAT = "float"
    W8A8 = "w8a8"
    W8A16 = "w8a16"
    W8A8S = "w8a8s"
    W8A8SC = "w8a8sc"


QUANT_W8A8_DESC_LIST = [QuantType.W8A8.upper(), QuantType.W8A8S.upper()]
