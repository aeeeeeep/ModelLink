# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib
from torch.nn import functional as F

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
    HAVE_PERSIST_LAYER_NORM = True
except ImportError:
    HAVE_PERSIST_LAYER_NORM = False

global fused_mix_prec_layer_norm_cuda
fused_mix_prec_layer_norm_cuda = None


class FusedLayerNormAffineFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, weight, bias, normalized_shape, eps):

    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    input_ = input.contiguous()
    weight_ = weight.contiguous()
    bias_ = bias.contiguous()
    output, mean, invvar = fused_mix_prec_layer_norm_cuda.forward_affine(
        input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
    ctx.save_for_backward(input_, weight_, bias_, mean, invvar)

    return output


  @staticmethod
  def backward(ctx, grad_output):

    input_, weight_, bias_, mean, invvar = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    grad_input, grad_weight, grad_bias \
      = fused_mix_prec_layer_norm_cuda.backward_affine(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        weight_, bias_, ctx.eps)

    return grad_input, grad_weight, grad_bias, None, None



class MixedFusedLayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, no_persist_layer_norm=True, sequence_parallel=False):
        super(MixedFusedLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(*normalized_shape))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):
        return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)