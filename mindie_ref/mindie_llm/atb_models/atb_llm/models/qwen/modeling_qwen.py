# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE

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

from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger


class QwenConfig(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=151936,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            emb_dropout_prob=0.0,
            attn_dropout_prob=0.0,
            layer_norm_epsilon=1e-6,
            initializer_range=0.02,
            max_position_embeddings=8192,
            scale_attn_weights=True,
            use_cache=True,
            bf16=False,
            fp16=False,
            fp32=False,
            kv_channels=128,
            rotary_pct=1.0,
            rotary_emb_base=10000,
            use_dynamic_ntk=True,
            use_logn_attn=True,
            use_flash_attn="auto",
            intermediate_size=22016,
            no_bias=True,
            tie_word_embeddings=False,
            use_cache_quantization=False,
            use_cache_kernel=False,
            softmax_in_fp32=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.emb_dropout_prob = emb_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.bf16 = bf16
        self.fp16 = fp16
        self.fp32 = fp32
        self.kv_channels = kv_channels
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.use_flash_attn = use_flash_attn
        self.no_bias = no_bias
        self.use_cache_quantization = use_cache_quantization
        self.use_cache_kernel = use_cache_kernel
        self.softmax_in_fp32 = softmax_in_fp32

        self.hidden_act = "silu"
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
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
        linear_names = [f'{prefix}.w1', f'{prefix}.w2']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_2'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, None)
        self.w2_w1 = load_column_multi(
            config,
            prefixes=[f"{prefix}.w2", f"{prefix}.w1"],  # gate_up_proj
            weights=weights,
            head_size=1,
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",  # down_proj
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

        linear_names = [f'{prefix}.c_attn']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_1'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, None)
        self.c_attn = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.c_attn",
            weights=weights,
            bias=True,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
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
        prefix = f"transformer.h.{layer_id}"
        self.attn = FlashQwenAttention(
            prefix=f"{prefix}.attn", config=config, weights=weights
        )
        self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        if self.attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.ln_1 = QwenRMSNorm(
                prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.ln_1 = QwenRMSNormBias(
                prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI]:
            self.ln_1 = QwenRMSNormWrapper(
                prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
            )
        else:
            raise AssertionError(f'self_attn.pack_type: {self.self_attn.pack_type} not supported')
        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16]:
            self.ln_2 = QwenRMSNorm(
                prefix=f"{prefix}.ln_2",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8]:
            self.ln_2 = QwenRMSNormBias(
                prefix=f"{prefix}.ln_2",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI]:
            self.ln_2 = QwenRMSNormWrapper(
                prefix=f"{prefix}.ln_2", weights=weights, eps=config.layer_norm_epsilon
            )
        else:
            raise AssertionError(f'mlp.pack_type: {self.mlp.pack_type} not supported')

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
            prefix="transformer.wte", weights=weights
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
            prefix="transformer.ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

        self.gradient_checkpointing = False

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads
