# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
import math
import os
from typing import Optional, List, Tuple

import torch
import torch.distributed
import torch_npu
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import (
    AttentionMask,
    TensorParallelHead,
)
from atb_llm.utils.log import logger, print_log

from .modeling_starcoder import StarcoderConfig, FlashStarcoderModel
from ...utils.data.weight_wrapper import AttnModuleNames, MlpModuleNames, WeightWrapper


class FlashStarcoderForCausalLM(torch.nn.Module):

    def __init__(self, config, weights):
        super().__init__()
        load_atb_speed()
        self.use_refactor = 1
        self.soc_info = NPUSocInfo()
        self.num_attention_heads = config.num_attention_heads
        if hasattr(config, 'num_key_value_heads'):
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_layers = config.num_layers
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_heads = config.head_num // self.tp_world_size  # 48
        self.max_seq_len_every_batch = config.seq_length
        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = config.multi_query_group_num
        self.max_position_embeddings = config.seq_length
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.quantize = config.quantize
        self.dtype = weights.dtype
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.device = weights.device
        # for ascend init
        self.init_ascend_operations(config)
        if self.use_refactor == 1:
            self.acl_encoder_operation_inputs = [None] * 8
            self.acl_decoder_operation_inputs = [None] * 8
        else:
            self.acl_encoder_operation_inputs = [None] * (8 + self.num_layers * 2)
            self.acl_decoder_operation_inputs = [None] * (8 + self.num_layers * 2)

        self.max_seq_leneqlen_tensor = torch.tensor([0], dtype=torch.int)
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()

        self.attention_mask_max_en = None
        self.attention_mask_max_de = None

        # self.attn_mask = AttentionMask.static(config.seq_length)
        self.max_base_len = 128
        self.attn_mask = AttentionMask.static(self.max_base_len, dtype=self.dtype)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.acl_param = None
        self.model = FlashStarcoderModel(config, weights)
        self.lm_head = TensorParallelHead.load_weight(
            config,
            prefix="lm_head",
            weights=weights,
            is_norm=True
        )

        self.placeholder = torch.ones(1, dtype=self.dtype, device="npu")
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int).to(self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).to(self.device)
        self.attn_mask_fake = self.attn_mask \
            .get_attn_mask(1, dtype=self.dtype, device="cpu") \
            .to(self.device)

    def init_kvcache(self, kv_cache):
        if self.use_refactor == 1:
            kcache_id = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
            vcache_id = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
            if kcache_id or vcache_id:
                k_caches, v_caches = map(list, zip(*kv_cache))
                print_log(self.tp_rank, logger.info, f"<<<<<<< ori {k_caches[0].shape=}")
                if self.soc_info.need_nz:
                    k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                    v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                    logger.info(f"<<<<<<<after transdata {k_caches[0].shape=}")
                self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
                self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
                self.ascend_kcache_id = id(kv_cache[0][0])
                self.ascend_vcache_id = id(kv_cache[0][1])
                print_log(self.tp_rank, logger.info,
                          f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")
        else:
            kvcache_status = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0]) \
                             or not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
            if kvcache_status:
                k_caches, v_caches = map(list, zip(*kv_cache))
                if self.soc_info.need_nz:
                    for i in range(self.num_layers):
                        torch_npu.npu_format_cast_(k_caches[i], 29)
                        torch_npu.npu_format_cast_(v_caches[i], 29)
                self.ascend_kcache_id = id(kv_cache[0][0])
                self.ascend_vcache_id = id(kv_cache[0][1])
                self.acl_encoder_operation_inputs[8: 8 + self.num_layers] = k_caches
                self.acl_encoder_operation_inputs[8 + self.num_layers: 8 + 2 * self.num_layers] = v_caches
                self.acl_decoder_operation_inputs[8: 8 + self.num_layers] = k_caches
                self.acl_decoder_operation_inputs[8 + self.num_layers: 8 + 2 * self.num_layers] = v_caches

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,  # position_ids
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
            **kwargs,
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices)
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        if self.use_refactor == 1:
            if lm_head_indices is not None:
                logits = logits[lm_head_indices]
            else:
                logits = logits

            return logits
        else:
            if lm_head_indices is not None:
                logits = logits.squeeze(0)[lm_head_indices]
            else:
                logits = logits.squeeze(0)
            return logits

    def init_ascend_operations(self, config: StarcoderConfig):
        if self.use_refactor == 1:
            self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("star_coder_PAQuantModel")
            self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("star_coder_PAQuantModel")
        else:
            self.acl_param_encoder = json.dumps({
                "layerNormEps": self.layer_norm_epsilon,
                "headNum": self.num_heads,
                "dk": self.head_dim,
                "kvHead": 1,
                "layerNum": self.num_layers,
                "rank": self.tp_rank,
                "rankSize": self.tp_world_size,
                "numHeadsPerPartition": self.num_key_value_heads,
                "isPrefill": True,
                "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),
            })
            self.acl_param_decoder = json.dumps({
                "layerNormEps": self.layer_norm_epsilon,
                "headNum": self.num_heads,
                "dk": self.head_dim,
                "kvHead": 1,
                "layerNum": self.num_layers,
                "rank": self.tp_rank,
                "rankSize": self.tp_world_size,
                "numHeadsPerPartition": self.num_key_value_heads,
                "isPrefill": False,
                "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),
            })
            self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("star_coder_PAModel")
            self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("star_coder_PAModel")
            self.acl_encoder_operation.set_param(self.acl_param_encoder)
            self.acl_decoder_operation.set_param(self.acl_param_decoder)

    def weight_format_cast(self, weight):
        if not self.soc_info.need_nz:
            return weight
        torch_npu.npu_format_cast_(weight, 29)
        return weight

    def get_weights(self):
        attn_module_names = AttnModuleNames(
            norm_name='input_layernorm',
            pack_name='self_attn.qkv',
            o_name='self_attn.o_proj'
        )
        mlp_module_names = MlpModuleNames(
            norm_name='post_attention_layernorm',
            pack_name='mlp.up_proj',
            down_name='mlp.down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_module_names, mlp_module_names)
        weight_wrapper.register_embedding(self.model.state_dict(), 'wte')
        weight_wrapper.register_embedding(self.model.state_dict(), 'wpe')
        for i in range(self.num_layers):
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

    def init_ascend_weight(self):

        if self.use_refactor == 1:
            self.ascend_weight, self.linear_type, self.pack_quant_config = self.get_weights()
            coder_param = {
                "isFA": False,
                "isBF16": self.dtype == torch.bfloat16,
                "isEmbeddingParallel": False,
                "isLmHeadParallel": True,
                "supportSwiGLU": False if self.soc_info.need_nz else True,
                "rank": self.tp_rank,
                "worldSize": self.tp_world_size,
                "backend": "hccl" if self.soc_info.need_nz else "lccl",
                "LayerNormEps": self.layer_norm_epsilon,
                "numAttentionHeadsPerRank": self.num_attention_heads,
                "hiddenSizePerAttentionHead": self.head_size,
                "numHiddenLayers": self.num_layers,
                "numKeyValueHeadsPerRank": self.num_key_value_heads,
                "packQuantType": self.pack_quant_config,
                "linearQuantType": self.linear_type,
                "layerNormEps": self.layer_norm_epsilon,
                "headNum": self.num_heads,
                "dk": self.head_dim,
                "kvHead": 1,
                "layerNum": self.num_layers,
                "rank": self.tp_rank,
                "rankSize": self.tp_world_size,
                "numHeadsPerPartition": self.num_key_value_heads,
            }
            encoder_param = {**coder_param, "isPrefill": True}
            decoder_param = {**coder_param, "isPrefill": False}
            self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
            self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

            self.acl_encoder_operation.set_weight(self.ascend_weight)
            self.acl_decoder_operation.set_weight(self.ascend_weight)
        else:
            weights = []
            weights.append(self.model.state_dict()["wte.weight"])
            weights.append(self.model.state_dict()["wpe.weight"])
            for i in range(self.num_layers):
                weights_t = []
                weights_layer = self.model.layers[i].state_dict()
                weights_t.append(weights_layer["input_layernorm.weight"])
                weights_t.append(weights_layer["input_layernorm.bias"])
                weights_t.append(self.weight_format_cast(weights_layer["self_attn.qkv.linear.weight"]))
                weights_t.append(weights_layer["self_attn.qkv.linear.bias"])
                weights_t.append(self.weight_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))
                weights_t.append(weights_layer["self_attn.o_proj.linear.bias"])
                weights_t.append(weights_layer["post_attention_layernorm.weight"])
                weights_t.append(weights_layer["post_attention_layernorm.bias"])
                weights_t.append(self.weight_format_cast(weights_layer["mlp.up_proj.linear.weight"]))
                weights_t.append(weights_layer["mlp.up_proj.linear.bias"])
                weights_t.append(self.weight_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
                weights_t.append(weights_layer["mlp.down_proj.linear.bias"])
                weights.extend(weights_t)
            weights.append(self.model.state_dict()["norm.weight"])
            weights.append(self.model.state_dict()["norm.bias"])
            print((self.state_dict()["lm_head.linear.weight"]).shape)
            weights.append(self.weight_format_cast(self.state_dict()["lm_head.linear.weight"]))

            self.ascend_weight = weights
            self.acl_encoder_operation.set_weight(weights)
            self.acl_decoder_operation.set_weight(weights)

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None):

        if is_prefill:
            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype,
                                                    kv_cache[0][0].device)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            if self.use_refactor == 1:
                self.acl_encoder_operation_inputs[0] = input_ids
                self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            else:
                self.acl_encoder_operation_inputs[0] = input_ids.unsqueeze(0)
                self.acl_encoder_operation_inputs[1] = position_ids.unsqueeze(0).to(torch.int64)
            self.acl_encoder_operation_inputs[2] = atten_mask
            self.acl_encoder_operation_inputs[3] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[4] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[5] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = lm_head_indices.to(torch.int64)
            self.acl_encoder_operation_inputs[7] = self.placeholder
            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            if self.use_refactor == 1:
                self.acl_decoder_operation_inputs[0] = input_ids
                self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            else:
                self.acl_decoder_operation_inputs[0] = input_ids.unsqueeze(0)
                self.acl_decoder_operation_inputs[1] = position_ids.unsqueeze(0).to(torch.int64)
            self.acl_decoder_operation_inputs[2] = self.attn_mask_fake
            self.acl_decoder_operation_inputs[3] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[4] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[5] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = self.lm_head_indices_fake
            self.acl_decoder_operation_inputs[7] = self.placeholder
            return self.acl_decoder_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill: bool
                                ):

        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]

        return acl_hidden_state
