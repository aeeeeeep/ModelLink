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

import math
from typing import Optional, List, Tuple

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorEmbedding,
    TensorParallelEmbedding,
    load_column_multi,
    paged_attn,
    flash_attn,
    reshape_and_cache
)

from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.utils.quantize.w8a8 import calc_linear_pack_type
from atb_llm.utils.log import logger


class Qwen2Config(PretrainedConfig):
    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )



class QwenRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states, residual


class QwenRMSNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class QwenRMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.ori = QwenRMSNorm(prefix, weights, eps)
        self.anti = QwenRMSNormBias(f'{prefix}.module', weights, eps)


class QwenMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )
        linear_names = [f'{prefix}.up_proj', f'{prefix}.gate_proj']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        elif weights.quantize == "smooth_quant":
            self.pack_type = PackType.ALL_W8A8
        else:
            self.pack_type = PackType.ALL_FP
        self.w2_w1 = load_column_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],  # gate_up_proj
            weights=weights,
            head_size=1,
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",  # down_proj
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )

    def forward(self, hidden_states):
        gate_up_states = self.w2_w1(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.c_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashQwenAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5

        # can support self.num_heads % weights.process_group.size() != 0
        
        linear_names = [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_1'
        if weights.quantize == 'w8a8':
            self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        elif weights.quantize == 'w8a16':
            self.pack_type = PackType.ALL_W8A16
        elif weights.quantize == "smooth_quant":
            self.pack_type = PackType.ALL_W8A8
        else:
            self.pack_type = PackType.ALL_FP
        
        # self.c_attn = TensorParallelColumnLinear.load_qkv(
        #     config,
        #     prefix=f"{prefix}.c_attn",
        #     weights=weights,
        #     bias=True,
        #     hidden_size=config.hidden_size,
        #     num_heads=config.num_attention_heads
        # )
        # self.c_attn = load_column_multi(
        #     config,
        #     prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],  # KQV
        #     weights=weights,
        #     head_size=1,
        # )

        self.c_attn = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            weights=weights,
            bias=True,
            dim=0
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

        self.prefix = prefix

    def forward(
            self,
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        qkv = self.c_attn(hidden_states)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_heads, self.head_size)

        self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)

        reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            paged_attn(
                attn_output,
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        return self.c_proj(attn_output.view(-1, self.num_heads * self.head_size))


class FlashQwenLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.attn = FlashQwenAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        if self.attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.ln_1 = QwenRMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.ln_1 = QwenRMSNormBias(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        else:
            self.ln_1 = QwenRMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.ln_2 = QwenRMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.ln_2 = QwenRMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            self.ln_2 = QwenRMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
            )

    def forward(
            self,
            hidden_states,
            residual,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        normed_hidden_states, res = self.ln_1(hidden_states, residual)

        # Self Attention
        attn_output = self.attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.ln_2(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashQwenModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.wte = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.h = nn.ModuleList(
            [
                FlashQwenLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = QwenRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads
