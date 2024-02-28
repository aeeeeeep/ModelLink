# coding=utf-8
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Aquila model."""
import math
from typing import List, Optional, Tuple, Union
import os
import json
import platform

import torch
from torch import nn
import torch_npu

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from atb_llm.models.aquila.v1_7b.config import AquilaConfig
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    PositionRotaryEmbedding,
    TensorParallelHead,
    AttentionMask
)
from atb_llm.utils.log import logger

_CONFIG_FOR_DOC = "AquilaConfig"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Aquila
class AquilaRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        AquilaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.LlamaMLP with Llama->Aquila
class AquilaMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.gate_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.gate_proj",
            weights=weights,
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.up_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.up_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                config.intermediate_size // weights.process_group.size()
        )
        self.act_fn = ACT2FN[config.hidden_act]


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->Aquila
class AquilaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        try:
            self.num_heads = math.ceil(self.num_heads / weights.process_group.size())
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        self.q_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.q_proj",
            weights=weights,
            bias=False,
        )
        self.k_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.k_proj",
            weights=weights,
            bias=False,
        )
        self.v_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.v_proj",
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )
        self.prefix = prefix


# Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with Llama->Aquila
class AquilaDecoderLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = AquilaAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = AquilaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = AquilaRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = AquilaRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )


# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->Aquila
class AquilaPreTrainedModel(PreTrainedModel):
    config_class = AquilaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AquilaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, AquilaModel):
            module.gradient_checkpointing = value


# Copied from transformers.models.llama.modeling_llama.LlamaModel with LLAMA->AQUILA,Llama->Aquila
class AquilaModel(AquilaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`AquilaDecoderLayer`]

    Args:
        config: AquilaConfig
    """

    def __init__(self, config: AquilaConfig, weights):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = TensorEmbedding(prefix="model.embed_tokens", weights=weights)
        self.layers = nn.ModuleList(
            [AquilaDecoderLayer(layer_id, config, weights, ) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = AquilaRMSNorm(prefix="model.norm", weights=weights, epsilon=config.rms_norm_eps)

        # for ascend
        self.training = False
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = config.num_key_value_heads // weights.process_group.size()

        self.soc_info = NPUSocInfo()
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.lm_head_weight = None
        self.is_prefill = True
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None

        self.ascend_atten_mask = AttentionMask.static(config.model_max_length)
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')

        self.batch_size = 0
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)

    def maybe_format_cast(self, tensor):
        """
        maybe_format_cast
        """
        if not self.soc_info.need_nz:  # transdata 会额外占资源
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def init_ascend_operations(self, config: AquilaConfig):
        self.acl_param_encoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": config.num_attention_heads // self.tp_world_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": True,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "isLmHeadParallel": True
        })
        self.acl_param_decoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": config.num_attention_heads // self.tp_world_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": False,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "isLmHeadParallel": True
        })
        logger.info(self.acl_param_encoder)
        logger.info(self.acl_param_decoder)
        logger.info("using aquila_7b_modeling_ascend")

        self.max_position_embeddings = config.max_position_embeddings
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("aquila_7b_PagedAttentionRopeModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("aquila_7b_PagedAttentionRopeModel")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.acl_operation_inputs = []
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.ascend_atten_mask = AttentionMask.static(config.max_position_embeddings)

    def init_ascend_weight(self):
        weights = [self.state_dict()["embed_tokens.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.layers[i].state_dict()
            weights_t.append(weights_layer["input_layernorm.weight"])
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.q_proj.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.k_proj.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.v_proj.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))
            weights_t.append(weights_layer["post_attention_layernorm.weight"])
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.gate_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.up_proj.linear.weight"]))
            weights.extend(weights_t)
            if self.soc_info.need_nz:
                del self.layers[i].self_attn
                del self.layers[i].mlp
            torch.npu.synchronize()
            peak_memory = torch_npu.npu.max_memory_allocated()
            logger.warning(f">>>>layer {i} peak_memory {peak_memory / 1024 / 1024} MB")
            torch.npu.synchronize()

        weights.append(self.state_dict()["norm.weight"])
        self.lm_head_weight = self.state_dict()["lm_head.linear.weight"]
        weights.append(self.maybe_format_cast(self.lm_head_weight))

        self.ascend_weight = weights
        self.acl_operation.set_weight(weights)

    def init_ascend_kvcache(self, kv_cache):
        kcache_id = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
            # k_cache shape [num_blocks, block_size, k_head_num, head_size] [36, 128, 40, 128]
            k_caches, v_caches = map(list, zip(*kv_cache))
            logger.debug(f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.debug(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            logger.warning(f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_s: int,
                                  lm_head_indices: Optional[torch.Tensor] = None):
        cos_embed, sin_embed = self.ascend_rotary_embedding.get_cos_sin_total(
            position_ids, max_s, torch.float32
        )
        if self.soc_info.need_nz:
            pad_maxs = math.ceil(max_s / 16) * 16
            atten_mask = self.ascend_atten_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
            atten_mask = self.transdata_operation.execute([atten_mask])[0]
        else:
            atten_mask = self.ascend_atten_mask.get_attn_mask(max_s, kv_cache[0][0].dtype, kv_cache[0][0].device)

        if self.is_prefill:  # prefill
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

        self.acl_operation_inputs = [
            input_ids,
            position_ids,
            cos_embed,
            sin_embed,
            atten_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            input_lengths.to(torch.int32),
            lm_head_indices if self.is_prefill else self.lm_head_indices_fake,
            self.place_holder
        ]
        for ind, item in enumerate(self.acl_operation_inputs):
            logger.debug(f"{ind} {item.device=}")
        return self.acl_operation_inputs

    def execute_ascend_operator(self,
                                acl_model,
                                input_ids: torch.Tensor,
                                position_ids: torch.Tensor,
                                is_prefill: bool,
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_s: int,
                                lm_head_indices: Optional[torch.Tensor] = None):
        acl_inputs = self.prepare_inputs_for_ascend(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            lm_head_indices)
        acl_param = json.dumps({
            "seqLen": input_lengths.tolist()
        })
        if self.is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,  #
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
    ):
        self.is_prefill = is_prefill
        self.batch_size = len(input_lengths)

        # add acl model
        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_ascend_kvcache(kv_cache)
        if is_prefill:
            operation = self.acl_encoder_operation
        else:
            operation = self.acl_decoder_operation

        hidden_states = self.execute_ascend_operator(
            operation,
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices)

        return tuple(v for v in [hidden_states] if v is not None)


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->AQUILA,Llama->Aquila
class FlashAquilaForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        load_atb_speed()
        self.model = AquilaModel(config, weights)
        self.soc_info = NPUSocInfo()
        logger.info(self.soc_info)
        # Initialize weights and apply final processing
        self.lm_head_weight = None
        self.parallel_lm_head = True
        self.model.parallel_lm_head = self.parallel_lm_head
        self.lm_head = (TensorParallelHead.load if self.parallel_lm_head else TensorParallelHead.load_weight)(
            config,
            prefix="lm_head",
            weights=weights,
            is_norm=True
        )

        self.num_heads = self.model.num_heads
        self.num_attention_heads = self.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.model.head_size
        self.num_key_value_heads = self.model.num_key_value_heads
        self.num_layers = config.num_hidden_layers

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,  #
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.model.lm_head_weight is None:
            if self.soc_info.need_nz:
                self.model.lm_head_weight = torch_npu.npu_format_cast(self.lm_head.weight.data, 29)
            self.model.lm_head_weight = self.lm_head.weight.data

        outputs = self.model(
            input_ids,  # input id, 拉平的
            is_prefill,  # prefill 阶段使用，不同prompt的offset
            kv_cache,  # kv cache,
            block_tables,  # 每个requests 所有的block tables
            slots,  # 每个requests 所有的slots
            input_lengths,  # 每个 request的k/v长度
            max_seq_len,  # 最长的request长度
            lm_head_indices  # prefill阶段使用，取的生成token的偏移
        )
        logits = outputs[0]
        return logits
