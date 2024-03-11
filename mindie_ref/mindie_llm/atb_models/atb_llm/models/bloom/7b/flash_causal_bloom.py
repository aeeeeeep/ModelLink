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

import os
import math
import time
import warnings
import json
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch_npu
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers import BloomConfig, BloomPreTrainedModel, AutoTokenizer, BloomTokenizerFast

from text_generation_server.utils.npu import load_atb_speed, NPUSocInfo
from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    TensorParallelHead,
    AttentionMask
)

from atb_llm.utils.log import logger

_CHECKPOINT_FOR_DOC = "bigscience/bloom-560m"
_CONFIG_FOR_DOC = "BloomConfig"
_TOKENIZER_FOR_DOC = "BloomTokenizerFast"

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
]


def cut_weights(state_dict, world_size, config, recuce_bias=False, cut_row_keys=('dense_h_to_4h',), cut_col_keys=('dense', 'dense_4h_to_h')):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in state_dict.items():
        key_short = key.split('.')[-2]
        key_type = key.split('.')[-1]

        if key_short == 'query_key_value':
            num_heads, head_dim = config.n_head, config.hidden_size // config.n_head
            dst_shape = list(tensor.shape)
            dst_shape[0] //= world_size

            tensor = tensor.view(num_heads, 3, head_dim, -1)
            tensor_list = torch.unbind(tensor, dim=1)
            chunk_tensor_list = [torch.chunk(item, world_size, dim=0) for item in tensor_list]
            cut_tensor_list = [torch.cat(item, 1).reshape(*dst_shape) for item in zip(*chunk_tensor_list)]
        else:
            if key_short in cut_row_keys:
                cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
            elif key_short in cut_col_keys:
                if key_type == "weight":
                    cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
                elif key_type == "bias":
                    if recuce_bias:
                        tensor = tensor / max(1, world_size)
                    cut_tensor_list = [tensor] * world_size
            else:
                cut_tensor_list = [tensor] * world_size

        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def print_rank_0(*args, **kwargs):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def record_npu_memory(name):
    print_rank_0(f"[{name.center(20,'*')}] [reserved:{torch.npu.memory_reserved()/1024/1024:.0f}M] [allocated: {torch.npu.memory_allocated()/1024/1024:.0f}M]")


def get_distributed_info():
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        rankSize = torch.distributed.get_world_size()
    else:
        rank = 0
        rankSize = 1
    return rank, rankSize


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    dtype = torch.float16
    batch_size, target_length = input_ids_shape
    mask = torch.full((target_length, target_length + past_key_values_length), torch.tensor(
        -10000, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask.masked_fill_((mask_cond.cpu() < (
        mask_cond + 1).view(mask.size(-1), 1).cpu()).npu(torch.npu.current_device()), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(
            target_length, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = mask[:, None, None, :].expand(
        batch_size, 1, tgt_length, src_length).to(torch.float16)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), -10000)


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.float32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    # for bloom, default is `left padding`, so `*attention_mask` is not necessary
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        slopes = slopes.reshape(world_size, -1)[rank, :].contiguous()
    else:
        world_size = 1
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads // max(1, world_size), 1, seq_length).to(dtype)


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
                _get_interleave_power_of_2(closest_power_of_2)
                + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    _future_mask = _future_mask.unsqueeze(0) + alibi
    new_future_mask = _future_mask.to(tensor)
    return new_future_mask[: tensor.shape[0] * attn_heads, :maxpos, :maxpos]


def _gen_alibi_mask(tensor, n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


def ntokens_trans_data_attention_mask(tensor, is_prefill=False):
    """
    prefill: [batch, head_num,max_s,max_s] -> [batch * head_num, maxS/16, maxS, 16]
    prefill: [4, 40, 1024, 1024]  ->  [160, 64, 1024, 16]
    max_s不够16整除的要pad 如[4,40,17,17] -> [4, 40, 17, 32] -> [160,2,17,16]

    decode: [batch, head_num,1,max_s] -> [batch * head_num, max_s/16, 16, 16]
    max_s不够16整除的要pad 如[1,40,1,17] -> [1, 40, 1, 32] -> [1, 40, 16, 32] ->[40,2,16,16]
    """
    logger.debug(f"shape of tensor in {is_prefill=} before transdata  is {tensor.shape}")
    nz_dim = 16
    if is_prefill:
        return torch_npu.npu_format_cast(tensor.view(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3] // max(1, nz_dim),
            nz_dim
        ).transpose(1, 2).contiguous(), 29)
    else:
        tensor = tensor.repeat(1, 1, nz_dim, 1)
        return torch_npu.npu_format_cast(tensor.view(
            tensor.shape[0] * tensor.shape[1],
            nz_dim,
            tensor.shape[3] // max(1, nz_dim),
            nz_dim
        ).transpose(1, 2).contiguous(), 29)


class FlashBloomForCausalLM(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config, weights):
        super().__init__(config)
        load_atb_speed()

        self.safe_weights = weights
        self.wt_device = weights.device
        self.config = config

        self.soc_info = NPUSocInfo()
        self.is_910b = not self.soc_info.need_nz
        self.is_float = True
        config.data_dtype = "fp16"
        config.model_path = r''
        config.model_max_length = 1024

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.n_head // self.tp_world_size
        self.hidden_size = config.hidden_size


        self.head_size = self.hidden_size // config.n_head
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = self.num_heads
        self.num_layers = self.num_hidden_layers

        # self.model_weights, self.float_layers = self.load_model(config)
        self.float_layers = list(range(self.num_hidden_layers))

        self.quant_param = {
            "qkvInputScale": [0] * self.num_hidden_layers,
            "qkvInputOffset": [0] * self.num_hidden_layers,
            "denseInputScale": [0] * self.num_hidden_layers,
            "denseInputOffset": [0] * self.num_hidden_layers,
            "selfLnInputScale": [0] * self.num_hidden_layers,
            "selfLnInputOffset": [0] * self.num_hidden_layers,
            "ffnOutInputScale": [0] * self.num_hidden_layers,
            "ffnOutInputOffset": [0] * self.num_hidden_layers
        }

        for layer_count in range(self.num_hidden_layers):
            if layer_count in self.float_layers:
                continue
            query_key_value_name = f"transformer.h.{layer_count}.self_attention.query_key_value"
            dense_name = f"transformer.h.{layer_count}.self_attention.dense"
            dense_h_to_4h_name = f"transformer.h.{layer_count}.mlp.dense_h_to_4h"
            dense_4h_to_h_name = f"transformer.h.{layer_count}.mlp.dense_4h_to_h"
            self.quant_param.get("qkvInputScale")[layer_count] = float(1 / self.input_scale_dict[query_key_value_name])
            self.quant_param.get("qkvInputOffset")[layer_count] = int(self.input_offset_dict[query_key_value_name])
            self.quant_param.get("denseInputScale")[layer_count] = float(1 / self.input_scale_dict[dense_name])
            self.quant_param.get("denseInputOffset")[layer_count] = int(self.input_offset_dict[dense_name])
            self.quant_param.get("selfLnInputScale")[layer_count] = float(1 / self.input_scale_dict[dense_h_to_4h_name])
            self.quant_param.get("selfLnInputOffset")[layer_count] = int(self.input_offset_dict[dense_h_to_4h_name])
            self.quant_param.get("ffnOutInputScale")[layer_count] = float(1 / self.input_scale_dict[dense_4h_to_h_name])
            self.quant_param.get("ffnOutInputOffset")[layer_count] = int(self.input_offset_dict[dense_4h_to_h_name])

        param_dict = {
            "layerNormEps": config.layer_norm_epsilon, "headNum": self.num_heads, "dk": config.hidden_size // config.n_head,
            "invNormFactorvarAttr": 1.0 / math.sqrt(config.hidden_size // config.n_head), "activationFuncType": 1,
            "layerNum": self.num_hidden_layers, "rank":self.tp_rank, "rankSize":self.tp_world_size, "floatLayers": self.float_layers,
            "backend": "lccl" if not self.soc_info.need_nz else "hccl"
            }
        param_dict.update(self.quant_param)
        
        self.acl_enc_model = torch.classes.ModelTorch.ModelTorch("bloom_7b_PagedAttentionModel")
        self.acl_dec_model = torch.classes.ModelTorch.ModelTorch("bloom_7b_PagedAttentionModel")
        

        param_dict['isPrefill'] = False
        self.param = json.dumps(param_dict)
        self.acl_dec_model.set_param(self.param)
        param_dict['isPrefill'] = True
        self.acl_enc_model.set_param(json.dumps(param_dict))
        self.first_run = True
        self.alibi_mask = None
        self.max_cache_pos = config.model_max_length
        self.n_head = config.n_head
        
        self.cache_k, self.cache_v = None, None 
        self.attention_mask_max = None
        self.attn_mask_in = None
        self.total_seqlen = 2048
        self.seq_len = 0
        
        self.bias_dtype = torch.int32
        self.deq_scale_dtype = torch.int64

        self.ascend_kcache_id = None
        self.ascend_vcache_id = None

        self.place_holder = torch.tensor([1], dtype=torch.float16).npu()
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()
        self.ascend_atten_mask = AttentionMask.static(config.model_max_length)

        # set weight
        self.weight_flag = True
        self.training = False
        if self.weight_flag:
            self.weights = self._init_weights_mine()
            self.acl_enc_model.set_weight(self.weights)
            self.acl_dec_model.set_weight(self.weights)
            torch.npu.empty_cache()
            self.weight_flag = False

    def maybe_transdata(self, x):
        return self.transdata_operation.execute([x])[0] if not self.is_910b else x

    def maybe_formatcast(self, x):
        return torch_npu.npu_format_cast(x, 29) if not self.is_910b else x

    def load_model(self, config):
        if config.data_dtype == "fp16":
            return self.load_model_fp16(config)
        elif config.data_dtype == "int8":
            return self.load_model_int8(config)
        else:
            raise Exception("inference dtype error!")
    
    def load_model_fp16(self, config):
        if self.world_size > 1:
            model_path = os.path.join(config.model_path, "part_model", str(self.rank))
        else:
            model_path = config.model_path
        
        model_weights_mapping = os.path.join(model_path, "pytorch_model.bin.index.json")

        with open(model_weights_mapping) as user_file:
            mapping_json = json.load(user_file)
        
        weight_files = list(set(mapping_json['weight_map'].values()))

        model_weights = {}
        for weight_file in weight_files:
            model_weights.update(torch.load(os.path.join(model_path, weight_file), map_location='cpu'))
        return model_weights, list(range(self.num_hidden_layers))

    def load_model_int8(self, config):
        if self.world_size > 1:
            model_path = os.path.join(config.model_path, "part_model", str(self.rank))
        else:
            model_path = config.model_path
        weight_list = [
            'quant_weight', 'bias', 'deq_scale',
            'input_scale', 'input_offset'
            ]
        
        weight_kwargs = {}
        for weight_name in weight_list:
            weight_kwargs[weight_name + '_dict'] = np.load(os.path.join(model_path, weight_name + '.npy'), allow_pickle=True).item()
        
        for weight_name, weight_values in weight_kwargs.items():
            setattr(self, weight_name, weight_values)
        
        float_weight_dict = torch.load(os.path.join(model_path, "/float_layers_weights.pt"))

        float_layers = []
        weights_keys = float_weight_dict.keys()
        for weights_key in weights_keys:
            key_split = weights_key.split('.')
            if 'h' in key_split and 'dense' in key_split:
                float_layers.append(int(key_split[2]))
        return float_weight_dict, list(set(float_layers))
        
    def _init_weights_mine(self):
        weights = [
            torch.chunk(self.safe_weights.get_tensor('word_embeddings.weight'), self.tp_world_size, dim=1)[self.tp_rank].clone().to(self.wt_device),
            self.safe_weights.get_tensor('word_embeddings_layernorm.weight').to(self.wt_device),
            self.safe_weights.get_tensor('word_embeddings_layernorm.bias').to(self.wt_device)
        ]
        for layer_num in range(self.num_hidden_layers):
            if layer_num in self.float_layers:
                weights_name_t = [
                    f'h.{layer_num}.input_layernorm.weight',
                    f'h.{layer_num}.input_layernorm.bias',
                    f'h.{layer_num}.self_attention.query_key_value.weight',
                    f'h.{layer_num}.self_attention.query_key_value.bias',
                    f'h.{layer_num}.self_attention.query_key_value.bias',
                    f'h.{layer_num}.self_attention.query_key_value.bias',
                    f'h.{layer_num}.self_attention.query_key_value.bias',
                    f'h.{layer_num}.self_attention.query_key_value.bias',
                    f'h.{layer_num}.self_attention.dense.weight',
                    f'h.{layer_num}.self_attention.dense.bias',
                    f'h.{layer_num}.self_attention.dense.bias',
                    f'h.{layer_num}.self_attention.dense.bias',
                    f'h.{layer_num}.self_attention.dense.bias',
                    f'h.{layer_num}.self_attention.dense.bias',
                    f'h.{layer_num}.post_attention_layernorm.weight',
                    f'h.{layer_num}.post_attention_layernorm.bias',
                    f'h.{layer_num}.mlp.dense_h_to_4h.weight',
                    f'h.{layer_num}.mlp.dense_h_to_4h.bias',
                    f'h.{layer_num}.mlp.dense_h_to_4h.bias',
                    f'h.{layer_num}.mlp.dense_h_to_4h.bias',
                    f'h.{layer_num}.mlp.dense_h_to_4h.bias',
                    f'h.{layer_num}.mlp.dense_h_to_4h.bias',
                    f'h.{layer_num}.mlp.dense_4h_to_h.weight',
                    f'h.{layer_num}.mlp.dense_4h_to_h.bias',
                    f'h.{layer_num}.mlp.dense_4h_to_h.bias',
                    f'h.{layer_num}.mlp.dense_4h_to_h.bias',
                    f'h.{layer_num}.mlp.dense_4h_to_h.bias',
                    f'h.{layer_num}.mlp.dense_4h_to_h.bias'
                ]
                weights_t = []
                for name in weights_name_t:
                    weight = self.safe_weights.get_tensor(name)
                    if "layernorm" not in name:
                        weight = cut_weights({name: weight}, self.tp_world_size, self.config)[self.tp_rank][name].clone().to(self.wt_device)
                        if "weight" in name:
                            weight = self.maybe_formatcast(weight)
                    else:
                        weight = weight.to(self.wt_device)
                    weights_t.append(weight)
                weights.extend(weights_t)
            else:
                query_key_value_name = f"transformer.h.{layer_num}.self_attention.query_key_value"
                dense_name = f"transformer.h.{layer_num}.self_attention.dense"
                dense_h_to_4h_name = f"transformer.h.{layer_num}.mlp.dense_h_to_4h"
                dense_4h_to_h_name = f"transformer.h.{layer_num}.mlp.dense_4h_to_h"
                weights_t = [
                    self.model_weights[f'{prefix}h.{layer_num}.input_layernorm.weight'].to(torch.float16).npu(),
                    self.model_weights[f'{prefix}h.{layer_num}.input_layernorm.bias'].to(torch.float16).npu(),

                    self.maybe_transdata(self.quant_weight_dict[query_key_value_name].to(torch.int8).npu()),
                    self.bias_dict[query_key_value_name].to(self.bias_dtype).npu(),
                    self.deq_scale_dict[query_key_value_name].to(self.deq_scale_dtype).npu(),

                    self.maybe_transdata(self.quant_weight_dict[dense_name].to(torch.int8).npu()), 
                    self.bias_dict[dense_name].to(self.bias_dtype).npu(),
                    self.deq_scale_dict[dense_name].to(self.deq_scale_dtype).npu(),

                    self.model_weights[f'{prefix}h.{layer_num}.post_attention_layernorm.weight'].to(torch.float16).npu(),
                    self.model_weights[f'{prefix}h.{layer_num}.post_attention_layernorm.bias'].to(torch.float16).npu(),

                    self.maybe_transdata(self.quant_weight_dict[dense_h_to_4h_name].to(torch.int8).npu()),
                    self.bias_dict[dense_h_to_4h_name].to(self.bias_dtype).npu(),
                    self.deq_scale_dict[dense_h_to_4h_name].to(self.deq_scale_dtype).npu(),

                    self.maybe_transdata(self.quant_weight_dict[dense_4h_to_h_name].to(torch.int8).npu()),
                    self.bias_dict[dense_4h_to_h_name].to(self.bias_dtype).npu(),
                    self.deq_scale_dict[dense_4h_to_h_name].to(self.deq_scale_dtype).npu()
                ]
                weights.extend(weights_t)

        weights.append(self.safe_weights.get_tensor('ln_f.weight').to(self.wt_device))
        weights.append(self.safe_weights.get_tensor('ln_f.bias').to(self.wt_device))
        # cut weights
        # weights.append(self.maybe_formatcast(self.safe_weights.get_tensor('word_embeddings.weight').to(self.wt_device)))
        weights.append(self.maybe_formatcast(torch.chunk(self.safe_weights.get_tensor('word_embeddings.weight'), self.tp_world_size, dim=1)[self.tp_rank].clone().to(self.wt_device)))
        return weights

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

        return combined_attention_mask
    
    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if past or past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            # if past[0][0].shape[0] == input_ids.shape[0]:
            #     past = self._convert_to_bloom_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past if past else past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def init_ascend_kvcache(self, kv_cache):
        if not self.ascend_kcache_id or self.ascend_kcache_id != kv_cache[0][0].data_ptr() \
                or not self.ascend_vcache_id or self.ascend_vcache_id != kv_cache[0][1].data_ptr():
            # k_cache.shape [num_blocks, block_size, k_head_num, head_size] [36, 128, 40, 128]
            self.kv_caches = kv_cache
            k_caches, v_caches = map(list, zip(*kv_cache))
            logger.info(f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.info(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_enc_model.set_kv_cache(k_caches, v_caches)
            self.acl_dec_model.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = kv_cache[0][0].data_ptr()
            self.ascend_vcache_id = kv_cache[0][1].data_ptr()
            logger.info(f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.training:
            slopes = torch.Tensor(_get_interleave(self.n_head))
            position_point = (torch.arange(seq_length_with_past) - seq_length_with_past + 1)
            position_point = (
                position_point.unsqueeze(0)
                .unsqueeze(0)
                .expand(self.n_head, seq_length_with_past, -1)
            )
            diag = torch.diag(position_point[0])
            position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(
                -1, -2
            )
            alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
            mask = _buffered_future_mask(
                tensor, seq_length_with_past, alibi, self.n_head
            )
        else:
            if self.first_run:
                self.first_run = False
                self.register_buffer(
                    "future_mask",
                    _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(
                        tensor
                    ),
                    persistent=False,
                )
            if seq_length_with_past > self.max_cache_pos:
                self.max_cache_pos = seq_length_with_past
                self.register_buffer(
                    "future_mask",
                    _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(
                        tensor
                    ),
                    persistent=False,
                )
            mask = self.future_mask[: self.n_head, :seq_length_with_past, :seq_length_with_past]
            logger.debug(f"{self.n_head=}")
            if self.tp_world_size > 1:
                mask = mask.chunk(self.tp_world_size, dim=0)
        return mask

    def generate_mask(self, max_s, kv_cache):
        """
        生成mask
        """
        pad_max_s = max_s
        if self.soc_info.need_nz:
            nz_dim = 16
            nz_pad = math.ceil(max_s / max(1, nz_dim)) * nz_dim - max_s
            pad_max_s = max_s + nz_pad
        attention_mask = self.ascend_atten_mask.get_attn_mask(pad_max_s, kv_cache[0][0].dtype, kv_cache[0][0].device)

        total_alibi_mask = self.get_alibi_mask(self.place_holder, pad_max_s)
        logger.debug(f"total_alibi_mask shape in {self.is_prefill=} is {total_alibi_mask[0].shape}")
        if self.tp_world_size > 1:
            total_alibi_mask = total_alibi_mask[self.tp_rank]
        if self.is_prefill:  # prefill
            attention_mask = attention_mask + total_alibi_mask  # [4, 40, 1024, 1024] [head_num,max_s,max_s]
            logger.debug(f"final attention_mask shape in {self.is_prefill=} is {attention_mask.shape}")
        else:
            attention_mask = total_alibi_mask  # [40, 1024, 1024] [head_num,max_s,max_s]
            attention_mask = attention_mask[:, -1:, :]
            logger.debug(f"final attention_mask shape in {self.is_prefill=} is {attention_mask.shape}")
        if self.soc_info.need_nz:
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.repeat(self.batch_size, 1, 1, 1)
            attention_mask = ntokens_trans_data_attention_mask(attention_mask, self.is_prefill)
            logger.debug(f"final attention_mask shape after transdata is {attention_mask.shape}")
        return attention_mask

    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        position_ids: Optional[torch.Tensor],  #
        is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
        block_tables: torch.Tensor,  # 每个requests 所有的block tables
        slots: torch.Tensor,  # 每个requests 所有的slots
        input_lengths: torch.Tensor,  # 每个 request的k/v长度
        max_seq_len: int,  # 最长的request长度
        lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        
        self.is_prefill = is_prefill
        self.batch_size = len(input_lengths)

        attention_mask = self.generate_mask(max_seq_len, kv_cache)

        self.init_ascend_kvcache(kv_cache)
        model = self.acl_enc_model if is_prefill else self.acl_dec_model
        
        if self.is_prefill:
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        else:
            lm_head_indices = torch.tensor([0], dtype=torch.int64, device=input_ids.device)
        self.acl_param_encoder = json.dumps(
            {"seqLen": input_lengths.cpu().tolist()}
            )
        
        acl_inputs = [
            input_ids,
            attention_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            input_lengths.to(torch.int32),
            self.place_holder,
            lm_head_indices
        ]

        acl_model_out = model.execute(acl_inputs, self.acl_param_encoder)

        return acl_model_out[1]