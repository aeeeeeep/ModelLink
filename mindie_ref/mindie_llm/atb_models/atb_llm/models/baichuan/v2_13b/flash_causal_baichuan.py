# Copyright (c) 2023, Baichuan Intelligent Technology. All rights reserved.
import os
import json
import math
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
import torch_npu
from atb_llm.models.baichuan.v2_13b.config import BaichuanConfig
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    TensorParallelHead,
    AttentionMask
)
from atb_llm.utils.log import logger
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from .modeling_baichuan import FlashBaichuanModel
from ....utils.data.weight_wrapper import AttnModuleNames, MlpModuleNames, WeightWrapper
from ...base.flash_causal_lm import FlashForCausalLM

from ....utils.layers import load_column_multi

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


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


class FlashBaichuanForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights):
        if not hasattr(config, 'max_position_embeddings'):
            config.max_position_embeddings = config.model_max_length
        self.use_refactor = True
        super().__init__(config, weights)
        del self.rotary_embedding
        self.model = FlashBaichuanModel(config, weights)
        self.lm_head_weight = None
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
            norm=config.vocab_size == 125696
            
        )
        self.ascend_atten_mask = AttentionMask.static(config.model_max_length)
        self.config = config  # for quantize
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')
        
        # for alibi
        self.training = False
        self.first_run = True
        self.max_cache_pos = config.model_max_length
        self.n_head = config.num_attention_heads
        # trans data
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)
        
    def init_ascend_weight(self):
        if not self.use_refactor:  # fp16
            torch.npu.synchronize()
            peak_memory = torch_npu.npu.max_memory_allocated()
            logger.warning(f">>>>before init ascend weights peak_memory {peak_memory / 1024 / 1024} MB")
            torch.npu.synchronize()
            weights = [self.model.state_dict()["embed_tokens.weight"]]
            for i in range(self.config.num_hidden_layers):
                weights_t = []
                weights_layer = self.model.layers[i].state_dict()
                weights_t.append(weights_layer["input_layernorm.weight"])
                weights_t.append(self.weight_format_cast(weights_layer["self_attn.W_pack.linear.weight"]))
                weights_t.append(self.weight_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))
                weights_t.append(weights_layer["post_attention_layernorm.weight"])
                weights_t.append(self.weight_format_cast(weights_layer["mlp.gate_up_proj.linear.weight"]))
                weights_t.append(self.weight_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
                weights.extend(weights_t)
                if self.soc_info.need_nz:
                    del self.model.layers[i].self_attn
                    del self.model.layers[i].mlp
                torch.npu.synchronize()
                peak_memory = torch_npu.npu.max_memory_allocated()
                logger.warning(f">>>>layer {i} peak_memory {peak_memory / 1024 / 1024} MB")
                torch.npu.synchronize()
            weights.append(self.model.state_dict()["norm.weight"])
            weights.append(self.lm_head_weight)
            self.ascend_weight = weights
            self.acl_encoder_operation.set_weight(weights)
            self.acl_decoder_operation.set_weight(weights)
        else:  # fp16 w8a8
            logger.info(f">>>> quant-{self.quantize}")
            self.ascend_weight, self.linear_type, self.pack_quant_config = self.get_weights()
            coder_param = {
                "isFA": False,
                "isBF16": False,
                "isEmbeddingParallel": True,
                "isLmHeadParallel": True,
                "supportSwiGLU": False if self.soc_info.need_nz else True,
                "rmsNormEps": self.config.rms_norm_eps,
                "numAttentionHeadsPerRank": self.config.num_attention_heads // self.tp_world_size,
                "hiddenSizePerAttentionHead": self.config.hidden_size // self.config.num_attention_heads,
                "numHiddenLayers": self.config.num_hidden_layers,
                "numKeyValueHeadsPerRank": self.num_key_value_heads,
                "rank": self.tp_rank,
                "worldSize": self.tp_world_size,
                "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),
                "packQuantType": self.pack_quant_config,
                "linearQuantType": self.linear_type,
            }
            encoder_param = {**coder_param, "isPrefill": True, "supportLcoc": False if self.soc_info.need_nz else True}
            decoder_param = {**coder_param, "isPrefill": False, "supportLcoc": False}
            self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
            self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
            
            logger.info(">>>> baichuan2_13b_PagedAttentionParam is inited.")
            self.acl_encoder_operation.set_weight(self.ascend_weight)
            self.acl_decoder_operation.set_weight(self.ascend_weight)
    
    def init_ascend_kvcache(self, kv_cache):
        kcache_id = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
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

    def init_ascend_operations(self, config: BaichuanConfig):
        """
        量化：加载模型
        浮点：加载参数、模型
        """
        if self.use_refactor:  # fp16、wa8a
            self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_PagedAttentionQuantModel")
            self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_PagedAttentionQuantModel")
        else:  # fp16
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
            logger.info("using flash_baichuan2_13b_modeling_ascend")
            
            self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_PagedAttentionModel")
            self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_PagedAttentionModel")
            self.acl_encoder_operation.set_param(self.acl_param_encoder)
            self.acl_decoder_operation.set_param(self.acl_param_decoder)

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_s: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  attention_mask=None):

        if self.is_prefill:  # prefill
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

        self.acl_operation_inputs = [
            input_ids,  # inputs_tensorids
            attention_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            input_lengths.to(torch.int32),
            lm_head_indices if self.is_prefill else self.lm_head_indices_fake,
            self.place_holder
        ]
        return self.acl_operation_inputs
    
    def generate_mask(self, max_s, kv_cache):
        """
        生成mask
        """
        pad_max_s = max_s
        if self.is_prefill:
            self.attention_mask = self.ascend_atten_mask.get_attn_mask(pad_max_s, kv_cache[0][0].dtype, kv_cache[0][0].device)
        total_alibi_mask = self.get_alibi_mask(self.place_holder, pad_max_s)  #
        if self.tp_world_size > 1:
            total_alibi_mask = total_alibi_mask[self.tp_rank]
        if self.is_prefill:  # prefill
            self.attention_mask = self.attention_mask + total_alibi_mask  # [4, 40, 1024, 1024] [head_num,max_s,max_s]
            logger.debug(f"final attention_mask shape in {self.is_prefill=} is {self.attention_mask.shape}")
        else:
            self.attention_mask = total_alibi_mask  # [40, 1024, 1024] [head_num,max_s,max_s]
            self.attention_mask = self.attention_mask[:, -1:, :]
            logger.debug(f"final attention_mask shape in {self.is_prefill=} is {self.attention_mask.shape}")
        
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
        
    def weight_format_cast(self, tensor):
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        return tensor

   
    def get_weights(self):
        attn_module_names = AttnModuleNames(
            norm_name='input_layernorm',
            pack_name='self_attn.W_pack',
            o_name='self_attn.o_proj'
        )
        mlp_module_names = MlpModuleNames(
            norm_name='post_attention_layernorm',
            pack_name='mlp.gate_up_proj',
            down_name='mlp.down_proj',
            gate_name='mlp.gate_proj',
            up_name='mlp.up_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_module_names, mlp_module_names)
        weight_wrapper.register_embedding(self.model.state_dict(), 'embed_tokens')
        for i in range(self.config.num_hidden_layers):
            layer = self.model.layers[i]
            layer_dict = layer.state_dict()
            weight_wrapper.register_layer(layer_dict, layer.self_attn.pack_type, layer.mlp.pack_type, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.state_dict(), 'norm')
        weight_wrapper.register_model_lmhead(self.state_dict(), 'lm_head')
        return weight_wrapper.weights, weight_wrapper.linear_type, weight_wrapper.pack_quant_type

    def execute_ascend_operator(self,
                                acl_model,
                                input_ids: torch.Tensor,
                                is_prefill: bool,
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_s: int,
                                lm_head_indices: Optional[torch.Tensor] = None,
                                attention_mask=None):
        acl_inputs = self.prepare_inputs_for_ascend(
            input_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
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
    ):
        if self.lm_head_weight is None:
            if self.config.vocab_size == 125696:
                logger.debug("baichuan2 13B normalize lm_head")
                self.lm_head_weight = nn.functional.normalize(self.state_dict()["lm_head.linear.weight"])
            else:
                self.lm_head_weight = self.state_dict()["lm_head.linear.weight"]
            if self.soc_info.need_nz:
                self.lm_head_weight.data = torch_npu.npu_format_cast(self.lm_head_weight.data, 29)
            self.model.lm_head_weight = self.lm_head_weight
        self.is_prefill = is_prefill
        self.batch_size = len(input_lengths)
        
        # generate self.attention_mask
        self.generate_mask(max_seq_len, kv_cache)
        
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
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,
            self.attention_mask)
        outputs = tuple(v for v in [hidden_states] if v is not None)
        logits = outputs[0]
        return logits