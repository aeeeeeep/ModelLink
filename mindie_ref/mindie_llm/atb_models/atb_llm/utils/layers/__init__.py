# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import List

import torch.distributed
from accelerate import init_empty_weights
from torch import nn
from torch.nn import functional as F

from atb_llm.common.log.logging import logger
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
    if not config.hidden_size % config.num_attention_heads == 0:
        logger.error('config.hidden_size % config.num_attention_heads != 0')
    if not config.num_attention_heads % weights.process_group.size() == 0:
        logger.error('config.num_attention_heads % weights.process_group.size()!= 0')

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize)
    )


def _load_column_multi(
        config, prefixes: List[str], weights, head_size, lm_head: bool = False, norm: bool = False
):
    if lm_head:
        weight = weights.get_multi_weights_col(prefixes, quantize=None, dim=0, gqa_size=head_size)
        weight = weight.npu()
        weight = torch.nan_to_num(weight if not norm else F.normalize(weight))  # 提前做norm  如果有nan则填充nan值
        linear = get_linear(weight, None, None)
    else:
        weight = weights.get_multi_weights_col(prefixes, quantize=config.quantize, dim=0, gqa_size=head_size)
        linear = get_linear(weight, None, config.quantize)

    if lm_head:
        if weights.process_group.size() != 1:
            return TensorParallelHead(linear, process_group=weights.process_group, should_gather=True)
        else:
            return TensorParallelHead(linear, process_group=weights.process_group, should_gather=False)
    else:
        return TensorParallelColumnLinear(linear)


def _load_row(config, prefix: str, weights, head_size):
    weight = weights.get_sharded(f"{prefix}.weight", dim=1, gqa_size=head_size)
    linear = get_linear(weight, None, quantize=config.quantize)
    return TensorParallelRowLinear(linear, process_group=weights.process_group)
