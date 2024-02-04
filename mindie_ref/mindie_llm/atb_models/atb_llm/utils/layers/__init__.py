# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import torch.distributed
from accelerate import init_empty_weights
from torch import nn
from torch.nn import functional as F

from atb_llm.utils.log import logger
from .attention import AttentionMask, flash_attn, paged_attn, reshape_and_cache
from .embedding.position_rotary_embedding import PositionRotaryEmbedding
from .embedding.tensor_embedding import TensorEmbedding, TensorParallelEmbedding
from .linear import get_linear, TensorParallelRowLinear, TensorParallelColumnLinear, TensorParallelHead, TensorHead


@classmethod
def load_layer_norm(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    bias = weights.get_tensor(f"{prefix}.bias")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = nn.Parameter(bias)
    return ln


@classmethod
def load_layer_norm_no_bias(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = None
    return ln


torch.nn.LayerNorm.load = load_layer_norm
torch.nn.LayerNorm.load_no_bias = load_layer_norm_no_bias


def _load_gqa(config, prefix: str, weights):
    hidden_size, num_attention_heads, process_group_size = config.hidden_size, config.num_attention_heads, weights.process_group.size()

    if not hidden_size % num_attention_heads == 0:
        logger.error(f'{hidden_size} % {num_attention_heads} != 0')
    if not num_attention_heads % process_group_size == 0:
        logger.error(f'{num_attention_heads} % {process_group_size} != 0')

    weight_prefixes = [f"{prefix}.{proj}" for proj in ["q_proj", "k_proj", "v_proj"]]
    weight = weights.get_multi_weights_col(prefixes=weight_prefixes, quantize=config.quantize, dim=0)

    return TensorParallelColumnLinear(get_linear(weight, bias=None, quantize=config.quantize))


<<<<<<< HEAD
def _load_column_multi(config, prefixes: List[str], weights, head_size, lm_head: bool = False, norm: bool = False):
    quantize = None if lm_head else config.quantize
    weight = weights.get_multi_weights_col(prefixes, quantize=quantize, dim=0, gqa_size=head_size)
=======
def load_column_multi(
        config, prefixes: List[str], weights, head_size, lm_head: bool = False, norm: bool = False
):
>>>>>>> cabb6e8 (change run_pa: add performance/profiling/max_postion_embedding)
    if lm_head:
        weight = weight.npu()
        weight = torch.nan_to_num(weight if not norm else F.normalize(weight))
    linear = get_linear(weight, None, quantize)

    process_group = weights.process_group
    should_gather = weights.process_group.size() != 1
    if lm_head:
        return TensorParallelHead(linear, process_group=process_group, should_gather=should_gather)
    else:
        return TensorParallelColumnLinear(linear)


def load_row(config, prefix: str, weights, head_size):
    weight = weights.get_sharded(f"{prefix}.weight", dim=1, gqa_size=head_size)
    linear = get_linear(weight, None, quantize=config.quantize)
    return TensorParallelRowLinear(linear, process_group=weights.process_group)
