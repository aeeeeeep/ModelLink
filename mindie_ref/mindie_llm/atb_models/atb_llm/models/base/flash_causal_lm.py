# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
from abc import abstractmethod
from typing import Optional, List, Tuple

import torch
import torch_npu
from transformers.configuration_utils import PretrainedConfig

from atb_llm.utils.log import logger, print_log
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import PositionRotaryEmbedding, AttentionMask


class FlashModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.gradient_checkpointing = False

    @abstractmethod
    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            cu_seqlen_prefill: Optional[torch.Tensor],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seqlen: int,
            lm_head_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass


class FlashForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        load_atb_speed()
        self.soc_info = NPUSocInfo()

        self.num_attention_heads = config.num_attention_heads
        if hasattr(config, 'num_key_value_heads'):
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_attention_heads
        if hasattr(config, 'rope_theta'):
            self.rope_theta = config.rope_theta
        else:
            self.rope_theta = 10000.0
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_layers = config.num_hidden_layers
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        print_log(self.tp_rank, logger.info, self.soc_info)

        # if num_key_value_heads is nondivisible 
        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) // self.tp_world_size

        self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                               device="cpu").to(weights.device)
        self.max_position_embeddings = config.max_position_embeddings
        self.quantize = config.quantize
        self.dtype = weights.dtype

        self.max_base_len = 128
        if self.soc_info.need_nz:
            self.attn_mask = AttentionMask.static(config.max_position_embeddings, dtype=self.dtype)
        else:
            self.attn_mask = AttentionMask.static(self.max_base_len, dtype=self.dtype)

        # for ascend init
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None

        self.in_tensor_length = 9
        self.acl_encoder_operation_inputs = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs = [None] * self.in_tensor_length

        self.device = weights.device
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int).to(self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).to(self.device)
        self.attn_mask_fake = self.attn_mask \
            .get_attn_mask(1, dtype=self.dtype, device="cpu") \
            .to(self.device)

        self.acl_param = None
        self.cos_embed = None
        self.sin_embed = None

    def weight_format_cast(self, tensor):
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        print_log(self.tp_rank, logger.info, f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    @abstractmethod
    def init_ascend_operations(self, config: PretrainedConfig):
        pass

    @abstractmethod
    def init_ascend_weight(self):
        pass

    def init_position_rotary_embedding(self, position_ids: torch.Tensor, max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
        if self.num_attention_heads == self.num_key_value_heads:
            self.cos_embed, self.sin_embed = self.rotary_embedding.get_cos_sin_cached_total(position_ids)
        else:
            self.cos_embed = self.rotary_embedding.get_cos_cached_total()
            self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_kvcache(self, kv_cache):
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

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None):
        self.init_position_rotary_embedding(position_ids, max_seq_len)
        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                atten_mask = atten_mask.view(1, pad_maxs, pad_maxs // 16, 16).transpose(1, 2)
                torch_npu.npu_format_cast_(atten_mask, 29)
            else:
                atten_mask = self.attn_mask.get_attn_mask(self.max_position_embeddings, kv_cache[0][0].dtype,
                                                                  kv_cache[0][0].device)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[2] = self.cos_embed
            self.acl_encoder_operation_inputs[3] = self.sin_embed
            if self.dtype == torch.bfloat16:
                self.acl_encoder_operation_inputs[4] = torch.where(atten_mask == -torch.inf, 1, atten_mask)
            else:
                self.acl_encoder_operation_inputs[4] = atten_mask
            self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[8] = lm_head_indices.to(torch.int64)
            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = self.cos_embed
            self.acl_decoder_operation_inputs[3] = self.sin_embed
            if self.dtype == torch.bfloat16:
                self.acl_decoder_operation_inputs[4] = torch.zeros(input_lengths.size(0),
                                                                    self.num_attention_heads,
                                                                    1, input_lengths.max(),
                                                                    dtype=self.dtype,
                                                                    device=input_ids.device)
            else:
                self.acl_decoder_operation_inputs[4] = self.attn_mask_fake
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[8] = self.lm_head_indices_fake
            return self.acl_decoder_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill):
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices)
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits
