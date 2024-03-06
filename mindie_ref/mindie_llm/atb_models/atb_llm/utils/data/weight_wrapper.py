# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
import torch_npu

from ..quantize.pack_type import PackType, LinearType
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
    def __init__(self, soc_info, tp_rank, attn_module_names, mlp_module_names):
        self.weights = []
        self.layer_linear_type = []
        self.linear_type = []
        self.soc_info = soc_info
        self.tp_rank = tp_rank
        self.placeholder = torch.zeros(1, dtype=torch.float16, device="npu")
        self.attn_module_names = attn_module_names
        self.mlp_module_names = mlp_module_names

    def weight_format_cast(self, tensor):
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        print_log(self.tp_rank, logger.info, f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def register_embedding(self, model_dict, embedding_name):
        self.weights.append(model_dict[f"{embedding_name}.weight"])

    def register_linear(self, layer_dict, linear_name, transpose=False):
        weight = layer_dict[f'{linear_name}.linear.weight']
        bias = layer_dict.get(f'{linear_name}.linear.bias', self.placeholder)

        if transpose:
            weight = weight.T.contiguous()
            bias = bias.T.contiguous()

        self.weights.append(self.weight_format_cast(weight))
        self.weights.append(self.weight_format_cast(bias))

    def register_norm(self, layer_dict, norm_name):
        self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.weight']))
        if f'{norm_name}.bias' in layer_dict:
            self.weights.append(self.weight_format_cast(layer_dict[f'{norm_name}.bias']))
        else:
            self.weights.append(self.placeholder)

    def register_layer_norm(self, layer_dict, norm_name):
        self.register_norm(layer_dict, f'{norm_name}')
        self.weights.extend([self.placeholder] * 2)

    def register_layer_norm_wrapper(self, layer_dict, norm_name):
        self.register_norm(layer_dict, f'{norm_name}.ori')
        self.register_norm(layer_dict, f'{norm_name}.anti')

    def register_layer_linear_pack_fp(self, layer_dict, norm_name, pack_linear_name, linear_type='attn'):
        self.register_layer_norm(layer_dict, norm_name)
        self.register_linear(layer_dict, pack_linear_name)
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 13)
            self.layer_linear_type.extend([LinearType.FP.value, LinearType.INVALID.value, LinearType.INVALID.value])
        else:
            self.weights.extend([self.placeholder] * 8)
            self.layer_linear_type.extend([LinearType.FP.value, LinearType.INVALID.value])

    def register_layer_linear_pack_w8a8(self, layer_dict, norm_name, pack_linear_name, pack_type, linear_type='attn'):
        if pack_type == PackType.ALL_W8A8:
            self.register_layer_norm(layer_dict, norm_name)
        else:
            self.register_layer_norm_wrapper(layer_dict, f'{norm_name}')
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.quant_bias']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.deq_scale']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.input_offset']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.input_scale']))
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 10)
            self.layer_linear_type.extend([LinearType.INT.value, LinearType.INVALID.value, LinearType.INVALID.value])
        else:
            self.weights.extend([self.placeholder] * 5)
            self.layer_linear_type.extend([LinearType.INT.value, LinearType.INVALID.value])

    def register_layer_linear_pack_w8a16(self, layer_dict, norm_name, pack_linear_name, linear_type='attn'):
        self.register_layer_norm(layer_dict, norm_name)
        self.register_linear(layer_dict, pack_linear_name, transpose=True)
        self.weights.append(self.placeholder)
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight_offset'].transpose(1, 0).contiguous()))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight_scale'].transpose(1, 0).contiguous()))
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 10)
            self.layer_linear_type.extend([LinearType.INT.value, LinearType.INVALID.value, LinearType.INVALID.value])
        else:
            self.weights.extend([self.placeholder] * 5)
            self.layer_linear_type.extend([LinearType.INT.value, LinearType.INVALID.value])

    def register_layer_linear_pack(self, layer_dict, norm_name, pack_linear_name, pack_type, linear_type='attn'):
        if pack_type == PackType.ALL_FP:
            self.register_layer_linear_pack_fp(layer_dict, norm_name, pack_linear_name, linear_type)
        elif pack_type == PackType.ALL_W8A16:
            self.register_layer_linear_pack_w8a16(layer_dict, norm_name, pack_linear_name, linear_type)
        else:
            self.register_layer_linear_pack_w8a8(layer_dict, norm_name, pack_linear_name, pack_type, linear_type)

    def register_layer_linear_pack_smoothquant(self, layer_dict, norm_name, pack_linear_name, pack_type,
                                               linear_type='attn'):
        self.register_layer_norm(layer_dict, norm_name)
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.weight']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.output_zeros']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.output_scales']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.act_zeros']))
        self.weights.append(self.weight_format_cast(layer_dict[f'{pack_linear_name}.linear.act_scales']))
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 10)
            self.layer_linear_type.extend([LinearType.INT.value, LinearType.INVALID.value, LinearType.INVALID.value])
        else:
            self.weights.extend([self.placeholder] * 5)
            self.layer_linear_type.extend([LinearType.INT.value, LinearType.INVALID.value])

    def register_layer_linear(self, layer_dict, linear_name, quantize_type):
        if layer_dict[f'{linear_name}.linear.weight'].dtype in [torch.float16, torch.bfloat16]:
            self.register_linear(layer_dict, linear_name)
            self.weights.extend([self.placeholder] * 3)
            self.layer_linear_type.append(LinearType.FP.value)
        elif quantize_type == 'w8a16':
            self.register_linear(layer_dict, linear_name, transpose=True)
            self.weights.append(self.placeholder)
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight_offset'].transpose(1, 0).contiguous()))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight_scale'].transpose(1, 0).contiguous()))
            self.layer_linear_type.append(LinearType.INT.value)
        elif quantize_type == 'smooth_quant':
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.act_scales']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.act_zeros']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.output_scales']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.output_zeros']))
            self.layer_linear_type.append(LinearType.INT.value)
        else:
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.weight']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.quant_bias']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.deq_scale']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.input_offset']))
            self.weights.append(self.weight_format_cast(layer_dict[f'{linear_name}.linear.input_scale']))
            self.layer_linear_type.append(LinearType.INT.value)

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
                    self.register_layer_norm(layer_dict, attn_module_names.norm_name)
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
                    self.register_layer_norm(layer_dict, mlp_module_names.norm_name)
                else:
                    self.register_layer_norm_wrapper(layer_dict, mlp_module_names.norm_name)
                self.register_layer_linear(layer_dict, mlp_module_names.gate_name, quantize_type)
                self.register_layer_linear(layer_dict, mlp_module_names.up_name, quantize_type)
        self.register_layer_linear(layer_dict, mlp_module_names.down_name, quantize_type)

    def register_model_norm(self, model_dict, norm_name):
        self.weights.append(model_dict[f'{norm_name}.weight'])

    def register_model_lmhead(self, model_dict, lmhead_name):
        self.weights.append(model_dict[f'{lmhead_name}.linear.weight'])
    
    def register_layer(self, layer_dict, attn_pack_type, mlp_pack_type, quantize_type):
        self.layer_linear_type.clear()
        self.register_layer_attn(layer_dict, attn_pack_type, quantize_type, self.attn_module_names)
        self.register_layer_mlp(layer_dict, mlp_pack_type, quantize_type, self.mlp_module_names)
        self.linear_type.append(self.layer_linear_type.copy())
