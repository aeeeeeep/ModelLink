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
"""PyTorch TELECHAT model."""

import json
import math
import os
from typing import Optional, List, Tuple

import torch
import torch.distributed
import torch_npu
from atb_llm.utils.initial import NPUSocInfo, load_atb_speed
from atb_llm.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorEmbedding,
    PositionRotaryEmbedding,
    AttentionMask,
    TensorParallelHead,
    load_column_multi,
    load_row
)
from atb_llm.utils.log import logger
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from .config import TelechatConfig
from atb_llm.utils.data.weight_wrapper import AttnModuleNames, MlpModuleNames, WeightWrapper
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM



# 量化开关，为量化总开关
RUN_QUANT_MODEL: bool = os.environ.get("RUN_QUANT_MODEL", "0") == "1"
# 量化权重路径
QUANT_WEIGHT_PATH = os.environ.get("QUANT_WEIGHT_PATH", "")
# 自定义最大输入输出长度，默认值2048
MAX_SEQ_LEN = os.environ.get("MAX_SEQ_LEN", 2048)
# 量化回退层选择
FLOAT_QUERY_LAYERS = []
FLOAT_KV_LAYERS = []
FLOAT_DOWN_LAYERS = [0, 1, 9, 25, 27]


lm_head_weight = None

class RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

class FlashTelechatAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config: TelechatConfig,
        weights,
    ):
        super().__init__()
        self.num_heads = config.n_head
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
            dim=self.head_dim, base=10000.0, device="cpu").to(weights.device)
        self.softmax_scale = self.head_dim ** -0.5

        try:
            self.num_heads = math.ceil(self.num_heads / weights.process_group.size())
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e

        self.query = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.query",
            weights=weights,
            bias=False,
        )
        self.key_value = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.key_value",
            weights=weights,
            bias=False,
        )
        self.dense = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense",
            weights=weights,
            bias=True,
        )
        self.prefix = prefix


class TelechatMLP(nn.Module):
    def __init__(self, prefix, config: TelechatConfig, weights):
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

        self.gate_and_up_bias = True
        if not RUN_QUANT_MODEL:
            gate_and_up_bias = False

        self.gate_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.gate_proj",
            weights=weights,
            bias=gate_and_up_bias,
        )
        self.up_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.up_proj",
            weights=weights,
            bias=gate_and_up_bias,
        )
        
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=True,
        )

        try:
            self.intermediate_size = (math.ceil(config.intermediate_size / weights.process_group.size()))
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e


class TelechatBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"h.{layer_id}"
        self.input_layernorm = RMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.layer_norm_epsilon
        )
        self.self_attention = FlashTelechatAttention(
            prefix=f"{prefix}.self_attention", config=config, weights=weights
        )
        self.mlp = TelechatMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.post_attention_layernorm = RMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )


class FlashTelechatModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()


        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        self.word_embeddings = TensorEmbedding(
            prefix="word_embeddings", weights=weights
        )

        # Transformer blocks
        self.h = nn.ModuleList(
            [
                TelechatBlock(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)]
        )
        # Final Layer Norm
        self.ln_f = RMSNorm(
            prefix="ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

        self.gradient_checkpointing = False

class FlashTelechatForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        load_atb_speed()
        self.config = config
        self.soc_info = NPUSocInfo()
        self.model = FlashTelechatModel(config, weights)
        self.parallel_lm_head = False

        if self.parallel_lm_head:
            self.lm_head = load_column_multi(
                config,
                prefixes=["word_embeddings"],
                weights=weights,
                head_size=1,
                lm_head=True,
                norm=self.confi..vocab_size == 125696
            )
        else:
            self.lm_head = TensorParallelHead.load_weight(
                config,
                prefix="word_embeddings",
                weights=weights,
                is_norm=True  # 不生效的配置
            )

        self.num_heads = config.n_head
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_key_value_heads = self.num_heads

        try:
            self.num_heads = math.ceil(self.num_heads / weights.process_group.size())
            self.num_key_value_heads = math.ceil(self.num_key_value_heads / weights.process_group.size())
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e

        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.ascend_rotary_embedding = PositionRotaryEmbedding.static(
            dim=self.head_size, base=10000.0, device="cpu").to(weights.device)
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')

        self.num_attention_heads = self.num_heads
        self.num_layers = config.num_hidden_layers
        self.is_prefill = True
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)
    
    def maybe_format_cast(self, tensor):
        if not self.soc_info.need_nz:  # transdata 会额外占资源
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        lgggrrnfofo(f"tr t tc_npggrch_npu.get_npu_ottmat(tensor)}")
        return tensor

    def init_ascend_operations(self, config):
        self.acl_param_encoder = json.dumps({
            "rmsNormEps":config.layer_norm_epsilon,
            "headNum": self.num_heads,
            "dk": self.head_size,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": True,
            "transposedWeight": True,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            # "isLmHeadParallel": self.parallel_lm_head
        })
        self.acl_param_decoder = json.dumps({
            "rmsNormEps": config.layer_norm_epsilon,
            "headNum": self.num_heads,
            "dk": self.head_size,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": False,
            "transposedWeight": True,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            # "isLmHeadParallel": self.parallel_lm_head
        })

        self.max_position_embeddings = config.max_position_embeddings
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("telechat_PAModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("telechat_PAModel")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.acl_operation_ipputs = []
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.ascend_atten_mask = AttentionMask.static(config.max_position_embeddings)
        self.ascend_atten_mask_fake = self.ascend_atten_mask.get_attn_mask(1, dtype=torch.float16, device="npu")


    def init_ascend_weight(self):
        weights = [self.model.state_dict()["word_embeddings.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = selfmmodel.h[i].state_dict()
            weights_t.append(self.maybe_format_cast(weights_layer["self_attention.query.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attention.key_value.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attention.dense.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attention.dense.linear.bias"]))
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.gate_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.up_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.down_proj.linear.bias"]))
            weights_t.append(weights_layer["input_layernorm.weight"])
            weights_t.append(weights_layer["post_attention_layernorm.weight"])
            weights.extend(weights_t)
            if self.soc_info.need_nz: # 释放内存，待检验
                del self.model.h[i].self_attention
                del self.model.h[i].mlp
        weights.append(self.model.state_dict()["ln_f.weight"])
        if self.soc_info.need_nz:
            del self.model
        self.lm_head_weight = self.state_dict()["lm_head.linear.weight"]
        weights.append(self.maybe_format_cast(self.lm_head_weight))

        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)

    def init_ascend_kvcache(self, kv_cache):
        kcache_id_exist = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id_exist = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kcache_id_exist or vcache_id_exist:
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
            position_ids, self.max_position_embeddings, torch.float32
        )
        if self.is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.ascend_atten_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.ascend_atten_mask.get_attn_mask(self.max_position_embeddings, kv_cache[0][0].dtype,
                                                                  kv_cache[0][0].device)
        else:
            atten_mask = self.ascend_atten_mask_fake
        if self.is_prefill:  # prefill
            lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)


        self.acl_operation_inputs = [
            input_ids,
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
                                input_ids: torch.Tensor,
                                position_ids: torch.Tensor,
                                is_prefill: bool,
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_s: int,
                                lm_head_indices: Optional[torch.Tensor] = None):
        acl_inputs = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                    block_tables, slots, input_lengths, max_s,
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
    ) -> torch.Tensor:
        self.is_prefill = is_prefill
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_ascend_kvcache(kv_cache)
        logits = self.execute_ascend_operator(input_ids, position_ids, is_prefill, kv_cache,
                                              block_tables, slots, input_lengths, max_seq_len, lm_head_indices)
        return logits