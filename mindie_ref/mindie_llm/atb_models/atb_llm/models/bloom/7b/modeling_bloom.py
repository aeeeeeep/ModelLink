# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch BLOOM model."""

import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.distributed
import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm
from torch.nn import functional as F

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers import BloomConfig, PreTrainedModel

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear
)

from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.utils.quantize.w8a8 import calc_linear_pack_type
from atb_llm.models.baichuan.v2_13b.config import BaichuanConfig


ADDEN_PREFIX = ""


class TensorParallelColumnEmbedding(nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_partial_sharded(f"{prefix}.weight", dim=1)

        self.process_group = weights.process_group
        self.reduce = reduce

        self.weight = nn.Parameter(weight)


class BloomAttention(nn.Module):
    def __init__(self, prefix, config: BloomConfig, weights):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.process_group = weights.process_group

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        process_group = weights.process_group
        if self.num_heads % process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {process_group.size()}"
            )
        self.num_heads = self.num_heads // process_group.size()
        self.query_key_value = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            bias=True,
        )

        # # to split like other MHA ??? 
        # self.query_key_value.linear.weight = torch.nn.Parameter(self.query_key_value.linear.weight.view(self.num_heads, 3, self.head_dim, self.hidden_size).transpose(0, 1).contiguous().view(-1, self.hidden_size), requires_grad=False)
        # if config.quantize:
        #     self.query_key_value.linear.quant_bias = torch.nn.Parameter(self.query_key_value.linear.quant_bias.view(self.num_heads, 3, self.head_dim).transpose(0, 1).contiguous().view(-1), requires_grad=False)
        #     self.query_key_value.linear.deq_scale = torch.nn.Parameter(self.query_key_value.linear.deq_scale.view(self.num_heads, 3, self.head_dim).transpose(0, 1).contiguous().view(-1), requires_grad=False)
        # else:
        #     self.query_key_value.linear.bias = torch.nn.Parameter(self.query_key_value.linear.bias.view(self.num_heads, 3, self.head_dim).transpose(0, 1).contiguous().view(-1), requires_grad=False)
    
        self.dense = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.dense", weights=weights, bias=True, bias_pre_add=True
        )
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        linear_names = [f'{prefix}.query_key_value']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        elif weights.quantize == "smooth_quant":
            self.pack_type = PackType.ALL_W8A8
        else:
            self.pack_type = PackType.ALL_FP


class BloomMLP(nn.Module):
    def __init__(self, prefix, config: BloomConfig, weights):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True
        )
        self.dense_4h_to_h = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True, bias_pre_add=True
        )
        self.gelu_impl = torch.nn.GELU(approximate="tanh")
        self.hidden_dropout = config.hidden_dropout

        linear_names = [f'{prefix}.dense_h_to_4h', f'{prefix}.dense_4h_to_h']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        #print(f"查看线性层的名字:{linear_names}")
        
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        elif weights.quantize == "smooth_quant":
            self.pack_type = PackType.ALL_W8A8
        else:
            self.pack_type = PackType.ALL_FP


class BloomBlock(nn.Module):
    def __init__(self, layer_id: int, config: BloomConfig, weights):
        super().__init__()

        prefix = f"h.{layer_id}"
        self.input_layernorm = LayerNorm.load(
            prefix=ADDEN_PREFIX + f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(
            prefix=ADDEN_PREFIX + f"{prefix}.self_attention", config=config, weights=weights
        )
        self.post_attention_layernorm = LayerNorm.load(
            prefix=ADDEN_PREFIX + f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.mlp = BloomMLP(prefix=ADDEN_PREFIX + f"{prefix}.mlp", config=config, weights=weights)
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.hidden_dropout = config.hidden_dropout


class FlashBloomModel(nn.Module):
    def __init__(self, config: BloomConfig, weights):
        super().__init__()
        if config.quantize:  # 量化后的模型会多一个transformer的前缀???
            global ADDEN_PREFIX
            ADDEN_PREFIX = "transformer."

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.word_embeddings = TensorParallelColumnEmbedding(
            prefix=ADDEN_PREFIX + "word_embeddings", weights=weights
        )

        self.word_embeddings_layernorm = LayerNorm.load(
            prefix=ADDEN_PREFIX + "word_embeddings_layernorm",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        # Transformer blocks
        self.h = nn.ModuleList(
            [
                BloomBlock(layer_id=layer_id, config=config, weights=weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        # Final Layer Norm
        self.ln_f = LayerNorm.load(
            prefix=ADDEN_PREFIX + "ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

        self.head_size = config.hidden_size // config.n_head
        self.num_heads = config.n_head
        self.num_key_value_heads = config.n_head