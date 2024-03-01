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


class AttnModuleNames:
    def __init__(self, norm_name, pack_name=None, q_name=None, k_name=None, v_name=None, o_name=None):
        self.norm_name = norm_name
        self.pack_name = pack_name
        self.q_name = q_name
        self.k_name = k_name
        self.v_name = v_name
        self.o_name = o_name


class MlpModuleNames:
    def __init__(self, norm_name, pack_name=None, gate_name=None, up_name=None, down_name=None):
        self.norm_name = norm_name
        self.pack_name = pack_name
        self.gate_name = gate_name
        self.up_name = up_name
        self.down_name = down_name


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
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.weight']))
        self.weights.append(self.placeholder)
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.bias']))

    def register_layer_norm_wrapper(self, layer_dict, norm_name):
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.ori.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.anti.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.anti.bias']))

    def register_layer_linear_pack_fp16(self, layer_dict, norm_name, pack_linear_name, linear_type='attn'):
        self.register_layer_norm(layer_dict, norm_name)
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight']))
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 14)
        else:
            self.weights.extend([self.placeholder] * 9)

    def register_layer_linear_pack_w8a8(self, layer_dict, norm_name, pack_linear_name, pack_type, linear_type='attn'):
        if pack_type == PackType.ALL_W8A8:
            self.register_layer_norm_bias(layer_dict, norm_name)
        else:
            self.register_layer_norm_wrapper(layer_dict, f'{norm_name}')
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.input_scale']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.input_offset']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.deq_scale']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.quant_bias']))
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 10)
        else:
            self.weights.extend([self.placeholder] * 5)

    def register_layer_linear_pack_w8a16(self, layer_dict, norm_name, pack_linear_name, linear_type='attn'):
        self.register_layer_norm(layer_dict, norm_name)
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight_scale']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight_offset']))
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 12)
        else:
            self.weights.extend([self.placeholder] * 7)

    def register_layer_linear_pack(self, layer_dict, norm_name, pack_linear_name, pack_type, linear_type='attn'):
        if pack_type == PackType.ALL_FP:
            self.register_layer_linear_pack_fp16(layer_dict, norm_name, pack_linear_name, linear_type)
        elif pack_type == PackType.ALL_W8A16:
            self.register_layer_linear_pack_w8a16(layer_dict, norm_name, pack_linear_name, linear_type)
        else:
            self.register_layer_linear_pack_w8a8(layer_dict, norm_name, pack_linear_name, pack_type, linear_type)

    def register_layer_linear_pack_smoothquant(self, layer_dict, norm_name, pack_linear_name, pack_type,
                                               linear_type='attn'):
        self.register_layer_norm_bias(layer_dict, norm_name)
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.act_scales']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.act_zeros']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.output_scales']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.output_zeros']))
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 10)
        else:
            self.weights.extend([self.placeholder] * 5)

    def register_layer_linear(self, layer_dict, linear_name, quantize_type):
        if layer_dict[f'{linear_name}.linear.weight'].dtype in [torch.float16, torch.bfloat16]:
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight']))
            self.weights.extend([self.placeholder] * 4)
            return
        if quantize_type == 'w8a16':
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight_scale']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight_offset']))
            self.weights.extend([self.placeholder] * 2)
        else:
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.input_scale']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.input_offset']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.deq_scale']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.quant_bias']))

    def register_layer_attn(self, layer_dict, pack_type, quantize_type, attn_module_names):
        if quantize_type == 'smooth_quant':
            self.register_layer_linear_pack_smoothquant(layer_dict,
                                                        attn_module_names.norm_name,
                                                        attn_module_names.pack_name,
                                                        pack_type,
                                                        'attn')
        else:
            if pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI,
                             PackType.ALL_W8A16]:
                self.register_layer_linear_pack(layer_dict,
                                                attn_module_names.norm_name,
                                                attn_module_names.pack_name,
                                                pack_type,
                                                'attn')
            else:
                if pack_type == PackType.MIX_W8A8:
                    self.register_layer_norm_bias(layer_dict, attn_module_names.norm_name)
                else:
                    self.register_layer_norm_wrapper(layer_dict, attn_module_names.norm_name)
                self.register_layer_linear(layer_dict, attn_module_names.q_name, quantize_type)
                self.register_layer_linear(layer_dict, attn_module_names.k_name, quantize_type)
                self.register_layer_linear(layer_dict, attn_module_names.v_name, quantize_type)
        self.register_layer_linear(layer_dict, attn_module_names.o_name, quantize_type)

    def register_layer_mlp(self, layer_dict, pack_type, quantize_type, mlp_module_names):
        if quantize_type == 'smooth_quant':
            self.register_layer_linear_pack_smoothquant(layer_dict,
                                                        mlp_module_names.norm_name,
                                                        mlp_module_names.pack_name,
                                                        pack_type,
                                                        'mlp')
        else:
            if pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI,
                             PackType.ALL_W8A16]:
                self.register_layer_linear_pack(layer_dict,
                                                mlp_module_names.norm_name,
                                                mlp_module_names.pack_name,
                                                pack_type,
                                                'mlp')
            else:
                if pack_type == PackType.MIX_W8A8:
                    self.register_layer_norm_bias(layer_dict, mlp_module_names.norm_name)
                else:
                    self.register_layer_norm_wrapper(layer_dict, mlp_module_names.norm_name)
                self.register_layer_linear(layer_dict, mlp_module_names.gate_name, quantize_type)
                self.register_layer_linear(layer_dict, mlp_module_names.up_name, quantize_type)
        self.register_layer_linear(layer_dict, mlp_module_names.down_name, quantize_type)

    def register_model_norm(self, model_dict, norm_name):
        self.weights.append(model_dict[f'{norm_name}.weight'])

    def register_model_lmhead(self, model_dict, lmhead_name):
        self.weights.append(model_dict[f'{lmhead_name}.linear.weight'])
