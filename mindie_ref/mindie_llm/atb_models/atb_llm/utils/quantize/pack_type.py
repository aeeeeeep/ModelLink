# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from enum import Enum
from .quant_type import QuantType, QUANT_W8A8_DESC_LIST


class PackType(Enum):
    ALL_FP = 1
    ALL_W8A8 = 2
    ALL_W8A8_ANTI = 3
    MIX_W8A8 = 4
    MIX_W8A8_ANTI = 5
    ALL_W8A16 = 6
    # ALL_W8A8S = 7
    # MIX_W8A8S = 8
    ALL_W8A8SC = 7
    MIX_W8A8SC = 8


def is_w8a8sc(type_desc):
    if type_desc == QuantType.W8A8SC.upper():
        return True
    else:
        return False


def calc_w8a8sc_linear_pack_type(weights, linear_names, pack_name=None):
    if pack_name:
        quant_desc = weights.quant_desc.get(f'{pack_name}.weight', None)
        if quant_desc == QuantType.W8A8SC.upper():
            return PackType.ALL_W8A8SC
        elif quant_desc == QuantType.FLOAT.upper():
            return PackType.ALL_FP

    linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
    is_w8a8sc_list = [is_w8a8sc(linear_desc) for linear_desc in linear_desces]

    is_any_w8a8sc = any(is_w8a8sc_list)
    if is_any_w8a8sc:
        if len(linear_names) == 1:
            return PackType.ALL_W8A8SC
        else:
            return PackType.MIX_W8A8SC
    return PackType.ALL_FP


def is_w8a8(type_desc):
    if type_desc in QUANT_W8A8_DESC_LIST:
        return True
    else:
        return False


def calc_w8a8_linear_pack_type(weights, linear_names, norm_name, pack_name=None):
    if pack_name:
        quant_desc = weights.quant_desc.get(f'{pack_name}.weight', None)
        if quant_desc in QUANT_W8A8_DESC_LIST:
            return PackType.ALL_W8A8
        elif quant_desc == QuantType.FLOAT:
            return PackType.ALL_FP

    linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
    norm_anti_desc = f'{norm_name}.module.weight'
    is_anti = True if norm_anti_desc in weights.quant_desc else False
    is_w8a8_list = [is_w8a8(linear_desc) for linear_desc in linear_desces]

    is_all_w8a8 = all(is_w8a8_list)
    is_any_w8a8 = any(is_w8a8_list)

    if is_anti:
        if is_all_w8a8:
            return PackType.ALL_W8A8_ANTI
        elif is_any_w8a8:
            return PackType.MIX_W8A8_ANTI
    else:
        if is_all_w8a8:
            return PackType.ALL_W8A8
        elif is_any_w8a8:
            return PackType.MIX_W8A8
    return PackType.ALL_FP


def calc_linear_pack_type(weights, linear_names, norm_name, pack_name=None):
    if weights.quantize in [QuantType.W8A8, QuantType.W8A8S]:
        pack_type = calc_w8a8_linear_pack_type(weights, linear_names, norm_name, pack_name)
    elif weights.quantize == QuantType.W8A16:
        pack_type = PackType.ALL_W8A16
    elif weights.quantize == "smooth_quant":
        pack_type = PackType.ALL_W8A8
    elif weights.quantize == QuantType.W8A8SC:
        pack_type = calc_w8a8sc_linear_pack_type(weights, linear_names, pack_name)
    else:
        pack_type = PackType.ALL_FP
    return pack_type


class LinearType(Enum):
    INVALID = -1
    FP = 0
    INT = 1
