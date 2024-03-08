# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import torch
import torch.distributed
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding
)
from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.utils.quantize.w8a8 import calc_linear_pack_type
from torch import nn
from transformers.configuration_utils import PretrainedConfig


class StarcoderConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=49152,
            n_embd=6144,
            n_head=48,
            n_inner=24576,
            n_layer=1,
            kv_channels=128,
            intermediate_size=24576,
            multi_query_group_num=1,
            num_attention_heads=48,
            hidden_act="gelu",
            n_positions=8192,
            initializer_range=0.02,
            layer_norm_epsilon=1e-05,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=0,
            tie_word_embeddings=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.head_num = n_head
        self.seq_length = n_positions
        self.hidden_size = n_embd
        self.kv_channels = 128
        self.intermediate_size = intermediate_size
        self.num_layers = n_layer
        self.multi_query_group_num = multi_query_group_num  # kv_head
        self.num_attention_heads = n_head
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class StarcoderLayerNormBias(nn.Module):
    def __init__(self, prefix, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)


class StarcoderLayerNormWrapper(nn.Module):
    def __init__(self, prefix, weights):
        super().__init__()
        # self.ori = StarcoderLayerNormBias(prefix, weights)
        self.anti = StarcoderLayerNormBias(f'{prefix}.module', weights)


class StarcoderMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        linear_names = [f"{prefix}.c_fc"]
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_2'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        else:
            self.pack_type = PackType.ALL_FP
        self.up_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.c_fc",
            weights=weights,
            bias=True,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=True,
            bias_pre_add=True
        )


class FlashStarcoderAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()

        linear_names = [f'{prefix}.c_attn']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_1'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        else:
            self.pack_type = PackType.ALL_FP
        self.qkv = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.c_attn",
            weights=weights,
            bias=True,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.multi_query_group_num
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=True,
            bias_pre_add=True
        )


class FlashStarcoderLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.h.{layer_id}"
        self.self_attn = FlashStarcoderAttention(
            prefix=f"{prefix}.attn", config=config, weights=weights
        )
        self.mlp = StarcoderMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16, PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.input_layernorm = StarcoderLayerNormBias(
                prefix=f"{prefix}.ln_1", weights=weights
            )
        else:
            self.input_layernorm = StarcoderLayerNormWrapper(
                prefix=f"{prefix}.ln_1", weights=weights
            )
        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16, PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.post_attention_layernorm = StarcoderLayerNormBias(
                prefix=f"{prefix}.ln_2", weights=weights,
            )
        else:
            self.post_attention_layernorm = StarcoderLayerNormWrapper(
                prefix=f"{prefix}.ln_2", weights=weights,
            )


class FlashStarcoderModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.wte = TensorEmbedding(
            prefix="transformer.wte", weights=weights
        )
        self.wpe = TensorEmbedding(
            prefix="transformer.wpe", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashStarcoderLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = StarcoderLayerNormBias(
            prefix="transformer.ln_f", weights=weights,
        )
