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
""" PyTorch InternLM model."""
import os
import math
import json
from typing import List, Optional, Tuple, Union
import threading, queue

import torch
import torch_npu
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.streamers import BaseStreamer
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, \
    replace_return_docstrings

from atb_llm.models.internlm.configuration_internlm import InternLMConfig
from atb_llm.utils.log.logging import logger
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    PositionRotaryEmbedding,
    TensorParallelHead,
    load_column_multi,
    load_row,
    AttentionMask
)

_CONFIG_FOR_DOC = "InternLMConfig"


def is_nd():
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version in [104, 220, 221, 222, 223, 224]


IS_ND = is_nd()
logger.info(f"IS_ND = {IS_ND}")


def get_rank_and_world_size():
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except:
        rank = 0
        world_size = 1
    return rank, world_size


RANK, WORLD_SIZE = get_rank_and_world_size()
logger.info(f"RANK = {RANK} | WORLD_SIZE = {WORLD_SIZE}")


def load_acl_transformer():
    """
    加载acl transformers
    :return:
    """
    acl_transformer_home_path = os.getenv("ATB_SPEED_HOME_PATH", "")
    if not acl_transformer_home_path or not os.path.exists(acl_transformer_home_path):
        raise RuntimeError("env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
    lib_path = os.path.join(acl_transformer_home_path, "lib/libatb_speed_torch.so")
    torch.classes.load_library(lib_path)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class InternLMRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        InternLMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class InternLMMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.gate_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.gate_proj",
            weights=weights,
            bias=False,
        )
        self.up_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.up_proj",
            weights=weights,
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                config.intermediate_size // weights.process_group.size()
        )
        self.act_fn = ACT2FN[config.hidden_act]


class InternLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_size = self.hidden_size // self.num_heads

        self.softmax_scale = self.head_size ** -0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )

        try:
            self.num_heads = math.ceil(self.num_heads / weights.process_group.size())
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e

        self.q_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.q_proj",
            weights=weights,
            bias=config.bias,
        )
        self.k_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.k_proj",
            weights=weights,
            bias=config.bias,
        )
        self.v_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.v_proj",
            weights=weights,
            bias=config.bias,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=config.bias,
        )
        self.prefix = prefix


class InternLMDecoderLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = InternLMAttention(prefix=f"{prefix}.self_attn", config=config, weights=weights)
        self.mlp = InternLMMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.input_layernorm = InternLMRMSNorm(prefix=f"{prefix}.input_layernorm", weights=weights,
                                               eps=config.rms_norm_eps)
        self.post_attention_layernorm = InternLMRMSNorm(prefix=f"{prefix}.post_attention_layernorm", weights=weights,
                                                        eps=config.rms_norm_eps)


INTERNLM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`InternLMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare InternLM Model outputting raw hidden-states without any specific head on top.",
    INTERNLM_START_DOCSTRING,
)
class InternLMPreTrainedModel(PreTrainedModel):
    config_class = InternLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLMDecoderLayer"]
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
        if isinstance(module, InternLMModel):
            module.gradient_checkpointing = value


class InternLMModel(InternLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLMDecoderLayer`]

    Args:
        config: InternLMConfig
    """
    _auto_class = "AutoModel"

    def __init__(self, config, weights):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = TensorEmbedding(prefix="model.embed_tokens", weights=weights)
        self.layers = nn.ModuleList(
            [InternLMDecoderLayer(layer_id=layer_id, config=config, weights=weights) for layer_id in
             range(config.num_hidden_layers)])
        self.norm = InternLMRMSNorm(prefix="model.norm", weights=weights, eps=config.rms_norm_eps)
        self.parallel_lm_head = True

        if self.parallel_lm_head:
            self.lm_head = TensorParallelHead.load(
                config,
                prefix="lm_head",
                weights=weights
            )
        else:
            self.lm_head = TensorParallelHead.load_weight(
                config,
                prefix="lm_head",
                weights=weights,
                is_norm=True  # 不生效的配置
            )
        self.gradient_checkpointing = False

        # for ascend
        self.training = False
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = math.ceil(config.num_attention_heads / weights.process_group.size())

        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_heads = self.num_heads // weights.process_group.size()

        self.soc_info = NPUSocInfo()
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None

        self.ascend_attn_mask = AttentionMask.static(config.max_position_embeddings)
        self.ascend_attn_mask_fake = self.ascend_attn_mask.get_attn_mask(1, dtype=torch.float16, device="npu")
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')

        # mindIE
        self.rotary_emb = PositionRotaryEmbedding.static(
            dim=self.head_size, base=config.rope_theta, device="cpu"
        ).to(weights.device)

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

    def init_ascend_operations(self, config: InternLMConfig):
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
        logger.info("using flash_internlm_20b_modeling_ascend")
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("internlm_20b_PagedAttentionModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("internlm_20b_PagedAttentionModel")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.num_layers = config.num_hidden_layers

        self.acl_operation_inputs = []
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")
        self.lm_head_weight = None
        self.is_prefill = True

    def ntoken_transdata(self, tensor):
        """
        prefill: [batch , head_num,max_s,max_s] -> [batch * head_num, maxS/16, maxS, 16]
        prefill: [4, 40, 1024, 1024]  ->  [160, 64, 1024, 16]
        max_s不够16整除的要pad 如[4,40,17,17] -> [4, 40, 17, 32] -> [160,2,17,16]

        decode: [batch,head_num,1,max_s] -> [batch * head_num, max_s/16, 16, 16]
        max_s不够16整除的要pad 如[1,40,1,17] -> [1, 40, 1, 32] -> [1, 40, 16, 32] ->[40,2,16,16]
        """
        return self.transdata_operation.execute(
            [tensor.view(tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3])]
        )[0]

    def init_ascend_weight(self):
        torch.npu.synchronize()
        peak_memory = torch_npu.npu.max_memory_allocated()
        logger.warning(f">>>>before init ascend weights peak_memory {peak_memory / 1024 / 1024} MB")
        torch.npu.synchronize()
        weights = [self.state_dict()["embed_tokens.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.layers[i].state_dict()
            weights_t.append(weights_layer["input_layernorm.weight"])
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.q_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.k_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.v_proj.linear.weight"]))
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

        # weights.append(self.lm_head_weight)
        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)

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

    def prepare_inputs_for_ascend(self,
                                  input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  attention_mask=None):
        cos_embed, sin_embed = self.rotary_emb.get_cos_sin_total(
            position_ids, max_seq_len, dtype=torch.float16
        )
        if self.is_prefill:  # prefill
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

        if self.is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(max_seq_len / 16) * 16
                attention_mask = self.ascend_attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype,
                                                                     kv_cache[0][0].device)
                attention_mask = attention_mask.view(1, pad_maxs, pad_maxs // 16, 16).transpose(1, 2)
                torch_npu.npu_format_cast_(attention_mask, 29)
            else:
                attention_mask = self.ascend_attn_mask.get_attn_mask(max_seq_len, kv_cache[0][0].dtype,
                                                                     kv_cache[0][0].device)
        else:
            attention_mask = self.ascend_attn_mask_fake

        self.acl_operation_inputs = [
            input_ids,
            position_ids,
            cos_embed,
            sin_embed,
            attention_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            input_lengths.to(torch.int32),
            lm_head_indices if self.is_prefill else self.lm_head_indices_fake,
            self.place_holder
        ]
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
                                max_seq_len: int,
                                lm_head_indices: Optional[torch.Tensor] = None,
                                attention_mask=None):
        acl_inputs = self.prepare_inputs_for_ascend(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,
            attention_mask)
        acl_param = json.dumps({
            "seqLen": input_lengths.tolist()
        })
        acl_model_out = acl_model.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
    ):
        self.is_prefill = is_prefill
        self.batch_size = len(input_lengths)

        logger.debug(f"{self.is_prefill=}")
        logger.debug(f"{input_ids.shape=}")
        logger.debug(f"{block_tables=}")
        logger.debug(f"{block_tables.shape=}")
        logger.debug(f"{slots=}")
        logger.debug(f"{slots.shape=}")
        logger.debug(f"{input_lengths=}")
        logger.debug(f"{input_lengths.shape=}")
        logger.debug(f"{max_seq_len=}")

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
            lm_head_indices, )

        return tuple(v for v in [hidden_states] if v is not None)


class FlashInternlmForCausalLM(InternLMPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config
        load_atb_speed()

        self.model = InternLMModel(config, weights)

        self.soc_info = NPUSocInfo()
        logger.info(self.soc_info)
        # Initialize weights and apply final processing
        self.lm_head_weight = None
        self.num_heads = self.model.num_heads
        self.num_key_value_heads = self.model.num_key_value_heads
        self.num_attention_heads = self.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.model.head_size
        self.num_layers = config.num_hidden_layers

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,  #
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids,  # input id, 拉平的
            position_ids,
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

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = []):
        prompt = ""
        for record in history:
            prompt += f"""<|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        return tokenizer([prompt], return_tensors="pt")

    @torch.no_grad()
    def chat(self,
             tokenizer,
             query: str,
             history: List[Tuple[str, str]] = [],
             streamer: Optional[BaseStreamer] = None,
             max_new_tokens: int = 1024,
             do_sample: bool = True,
             temperature: float = 0.8,
             top_p: float = 0.8,
             **kwargs):
        inputs = self.build_inputs(tokenizer, query, history)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.generate(**inputs,
                                streamer=streamer,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                                **kwargs)
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<eoa>")[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(self,
                    tokenizer,
                    query: str,
                    history: List[Tuple[str, str]] = [],
                    max_new_tokens: int = 1024,
                    do_sample: bool = True,
                    temperature: float = 0.8,
                    top_p: float = 0.8,
                    **kwargs):
        """
        Return a generator in format: (response, history)
        Eg.
        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')])
        ('你好，有什么可以帮助您的吗？', [('你好', '你好，有什么可以帮助您的吗？')])
        """

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ""
                self.received_inputs = False
                self.queue.put((self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                token = self.tokenizer.decode([value[-1]], skip_special_tokens=True)
                if token.strip() != "<eoa>":
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()


@add_start_docstrings(
    """
    The InternLM Model transformer with a sequence classification head on top (linear layer).

    [`InternLMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    INTERNLM_START_DOCSTRING,
)
class InternLMForSequenceClassification(InternLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
