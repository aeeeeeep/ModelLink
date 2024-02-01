# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from functools import partial

import torch
from torch import nn


def quantize_weight_per_channel_absmax(w, n_bits=8):
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8LinearStatic(nn.Module):
    def __init__(self, weight, weight_scales, act_scales, weight_zeros=None, act_zeros=None):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer('weight', weight.to(torch.int8))
        self.weight_quant_name = 'per_channel'

        self.act_quant_name = 'per_tensor'
        self.register_buffer('act_scales', act_scales.reshape(1).to(torch.float16))

        if act_zeros:
            self.register_buffer('act_zeros', act_zeros.to(torch.int8), requires_grad=False)
        else:
            self.register_buffer('act_zeros', torch.tensor([], dtype=torch.int8))

        self.weight_quant_name = 'per_channel'
        output_scales = torch.frombuffer((torch.mul(weight_scales, act_scales)).to(torch.float32).numpy().tobytes(),
                                         dtype=torch.int32)

        self.register_buffer('output_scales', output_scales.to(torch.int64))

        if weight_zeros:
            self.register_buffer('output_zeros', weight_zeros.to(torch.int32), requires_grad=False)
        else:
            self.output_zeros = None

        self.output_quant_name = 'per_channel'

        self.losses = None
        self.weight = None
        self.bias = None

    def __repr__(self):
        return (f'W8A8LinearStatic({self.in_features}, {self.out_features}, '
                f'bias={self.act_zeros is not None}, weight_quant={self.weight_quant_name}, '
                f'act_quant={self.act_quant_name}, output_quant={self.output_quant_name})')

    def act_quant(self, x, n_bits=8):
        q_max = 2 ** (n_bits - 1) - 1
        x.div_(self.act_scales).round_().add_(self.act_zeros)
        return x

    def output_quant(self, x, n_bits=8):
        q_max = 2 ** (n_bits - 1) - 1
        x.add_(self.output_zeros).mul_(self.output_scales)
        return x

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    def to(self, *args, **kwargs):
        super(W8A8LinearStatic, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self


class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', quantize_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def __repr__(self):
        return (f'W8A8Linear({self.in_features}, {self.out_features}, '
                f'bias={self.bias is not None}, weight_quant={self.weight_quant_name}, '
                f'act_quant={self.act_quant_name}, output_quant={self.output_quant_name})')

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False):
        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant,
            quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y
