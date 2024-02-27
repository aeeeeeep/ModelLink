# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn

from .pack_type import PackType


def is_w8a8(type_desc):
    if type_desc == 'W8A8':
        return True
    else:
        return False


def calc_attn_pack_type(prefix, weights):
    q_desc = weights.w8a8_desc[f'{prefix}.q_proj.weight']
    k_desc = weights.w8a8_desc[f'{prefix}.k_proj.weight']
    v_desc = weights.w8a8_desc[f'{prefix}.v_proj.weight']
    layer_prefix = '.'.join(prefix.split('.')[:-1])
    is_anti = True if f'{layer_prefix}.input_layernorm.module.weight' in weights.w8a8_desc else False
    is_q_w8a8 = is_w8a8(q_desc)
    is_k_w8a8 = is_w8a8(k_desc)
    is_v_w8a8 = is_w8a8(v_desc)

    if is_q_w8a8 and is_k_w8a8 and is_v_w8a8:
        if is_anti:
            return PackType.ALL_ANTI
        else:
            return PackType.ALL_INT
    elif not is_q_w8a8 and not is_k_w8a8 and not is_v_w8a8:
        return PackType.ALL_FP
    elif is_anti:
        return PackType.MIX_FP_ANTI
    else:
        return PackType.MIX_FP_INT


def calc_mlp_pack_type(prefix, weights):
    up_desc = weights.w8a8_desc[f'{prefix}.up_proj.weight']
    gate_desc = weights.w8a8_desc[f'{prefix}.gate_proj.weight']
    layer_prefix = '.'.join(prefix.split('.')[:-1])
    is_anti = True if f'{layer_prefix}.post_attention_layernorm.module.weight' in weights.w8a8_desc else False
    is_up_w8a8 = is_w8a8(up_desc)
    is_gate_w8a8 = is_w8a8(gate_desc)

    if is_up_w8a8 and is_gate_w8a8:
        if is_anti:
            return PackType.ALL_ANTI
        else:
            return PackType.ALL_INT
    elif not is_up_w8a8 and not is_gate_w8a8:
        return PackType.ALL_FP
    elif is_anti:
        return PackType.MIX_FP_ANTI
    else:
        return PackType.MIX_FP_INT


class W8A8LinearStatic(nn.Module):
    def __init__(self, weight, deq_scale, input_scale, quant_bias=None, input_offset=None):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer('weight', weight.to(torch.int8))
        self.weight_quant_name = 'per_channel'

        self.act_quant_name = 'per_tensor'
        self.register_buffer('input_scale', input_scale.reshape(1).to(torch.float16))

        if input_offset is not None:
            self.register_buffer('input_offset', input_offset)
        else:
            self.register_buffer('input_offset', torch.tensor([], dtype=torch.int8))

        self.weight_quant_name = 'per_channel'

        deq_scale = torch.frombuffer(deq_scale.to(torch.float32).numpy(), dtype=torch.int32)
        self.register_buffer('deq_scale', deq_scale.to(torch.int64))

        if quant_bias is not None:
            self.register_buffer('quant_bias', quant_bias.to(torch.int32))
        else:
            self.quant_bias = None

        self.output_quant_name = 'per_channel'
