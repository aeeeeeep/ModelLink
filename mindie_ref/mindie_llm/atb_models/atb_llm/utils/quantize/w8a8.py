# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn

# from .pack_type import PackType
# from .quant_type import QuantType
#
#
# QUANT_W8A8_SUPPORT_LIST = [QuantType.W8A8.upper(), QuantType.W8A8S.upper()]
#
# def is_w8a8(type_desc):
#     if type_desc in QUANT_W8A8_SUPPORT_LIST:
#         return True
#     else:
#         return False
#
#
# def calc_w8a8_linear_pack_type(weights, linear_names, norm_name):
#     linear_desces = [weights.quant_desc[f'{linear_name}.weight'] for linear_name in linear_names]
#     norm_anti_desc = f'{norm_name}.module.weight'
#     is_anti = True if norm_anti_desc in weights.quant_desc else False
#     is_w8a8_list = [is_w8a8(linear_desc) for linear_desc in linear_desces]
#
#     is_all_w8a8 = all(is_w8a8_list)
#     is_any_w8a8 = any(is_w8a8_list)
#
#     if is_anti:
#         if is_all_w8a8:
#             return PackType.ALL_W8A8_ANTI
#         elif is_any_w8a8:
#             return PackType.MIX_W8A8_ANTI
#     else:
#         if is_all_w8a8:
#             return PackType.ALL_W8A8
#         elif is_any_w8a8:
#             return PackType.MIX_W8A8
#     return PackType.ALL_FP


class W8A8LinearStatic(nn.Module):
    def __init__(self, weight, deq_scale, input_scale, quant_bias=None, input_offset=None):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer('weight', weight.to(torch.int8))

        self.act_quant_name = 'per_tensor'
        self.register_buffer('input_scale', input_scale.to(torch.float16))

        if input_offset is not None:
            self.register_buffer('input_offset', input_offset.to(torch.int8))
        else:
            self.register_buffer('input_offset', torch.tensor([], dtype=torch.int8))

        self.weight_quant_name = 'per_channel'

        self.register_buffer('deq_scale', deq_scale)

        if quant_bias is not None:
            self.register_buffer('quant_bias', quant_bias)
        else:
            self.quant_bias = None

        self.output_quant_name = 'per_channel'
