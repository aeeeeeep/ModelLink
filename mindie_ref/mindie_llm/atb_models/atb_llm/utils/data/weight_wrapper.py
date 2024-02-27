# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
import torch_npu

from ..quantize.pack_type import PackType
from ..log import logger, print_log


def weight_format_cast(tensor, soc_info):
    if not soc_info.need_nz:
        return tensor
    torch_npu.npu_format_cast_(tensor, 29)
    return tensor


class WeightWrapper:
    def __init__(self, soc_info, tp_rank):
        self.weights = []
        self.soc_info = soc_info
        self.tp_rank = tp_rank
        self.placeholder = torch.zeros(1, dtype=torch.float16, device="npu")

    def weight_format_cast(self, tensor):
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        print_log(self.tp_rank, logger.info, f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def register_embedding(self, model_dict, embedding_name):
        self.weights.append(model_dict[f"{embedding_name}.weight"])

    def register_layer_norm(self, layer_dict, norm_name):
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.weight']))
        self.weights.extend([self.placeholder] * 2)  # for anti

    def register_layer_norm_bias(self, layer_dict, norm_name):
        self.weights.append(self.placeholder)
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.bias']))

    def register_layer_norm_wrapper(self, layer_dict, norm_name):
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.ori.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.anti.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.anti.bias']))

    def register_layer_linear_pack_fp16(self, layer_dict, norm_name, pack_linear_name):
        self.register_layer_norm(layer_dict, norm_name)
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight']))
        self.weights.extend([self.placeholder] * 19)

    def register_layer_linear_pack_w8a8(self, layer_dict, norm_name, pack_linear_name, pack_type):
        if pack_type == PackType.ALL_INT:
            self.register_layer_norm_bias(layer_dict, norm_name)
        else:
            self.register_layer_norm_wrapper(layer_dict, f'{norm_name}')
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.input_scale']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.input_offset']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.deq_scale']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.quant_bias']))
        self.weights.extend([self.placeholder] * 10)  # 8

    def register_layer_linear_pack(self, layer_dict, norm_name, pack_linear_name, pack_type):
        if pack_type == PackType.ALL_FP:
            self.register_layer_linear_pack_fp16(layer_dict, norm_name, pack_linear_name)
        else:
            self.register_layer_linear_pack_w8a8(layer_dict, norm_name, pack_linear_name, pack_type)

    def register_layer_linear(self, layer_dict, linear_name):
        if layer_dict[f'{linear_name}.linear.weight'].dtype in [torch.float16, torch.bfloat16]:
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight']))
            self.weights.extend([self.placeholder] * 4)
        else:
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.input_scale']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.input_offset']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.deq_scale']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.quant_bias']))

    def register_model_norm(self, model_dict, norm_name):
        self.weights.append(model_dict[f'{norm_name}.weight'])

    def register_model_lmhead(self, model_dict, lmhead_name):
        self.weights.append(model_dict[f'{lmhead_name}.linear.weight'])
