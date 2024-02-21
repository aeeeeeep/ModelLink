# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from abc import abstractmethod
from typing import Optional, List, Tuple, Union

import math
import torch
import torch_npu
from torch.nn import CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from atb_llm.utils.log import logger
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import PositionRotaryEmbedding


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class Model(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.gradient_checkpointing = False

    @abstractmethod
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            max_s: Optional[int] = None,
            lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass


class CausalLM(PreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        load_atb_speed()
        self.soc_info = NPUSocInfo()
        logger.info(self.soc_info)

        self.num_attention_heads = config.num_attention_heads
        if hasattr(config, 'num_key_value_heads'):
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_layers = config.num_hidden_layers
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = self.num_key_value_heads // self.tp_world_size

        self.batch_num = 0
        self.mask_full = None
        self.mask_inc = None

        self.ascend_rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0,
                                                                      device="cpu").to(weights.device)
        self.max_position_embeddings = config.max_position_embeddings
        self.quantize = config.quantize

        self.init_ascend_operations(config)
        self.ascend_weight = []

        self.token_offset = None
        self.seq_len_encoder = None
        self.seq_len_decoder = None
        self.k_cache = None
        self.v_cache = None
        self.past_key_values_length = 0
        self.nz_dim = 16

    def weight_format_cast(self, tensor):
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    @abstractmethod
    def init_ascend_operations(self, config: PretrainedConfig):
        pass

    @abstractmethod
    def init_ascend_weight(self):
        pass

    def init_kvcache(self, input_ids, past_key_value):
        batch_size, _ = input_ids.shape

        if batch_size != self.batch_num:
            self.batch_num = batch_size
            self.token_offset = torch.full((self.batch_num,), 0, dtype=torch.int32, device=input_ids.device)
            self.seq_len_encoder = torch.full((self.batch_num,), 1, dtype=torch.int32, device=input_ids.device)
            self.seq_len_decoder = torch.full((self.batch_num,), 1, dtype=torch.int32, device=input_ids.device)
            self.acl_param_seq_len_decoder = [1] * self.batch_num
            self.mask_full = torch.zeros((self.batch_num, self.max_position_embeddings, self.max_position_embeddings),
                                         dtype=torch.half, device=input_ids.device)

        if past_key_value:
            self.k_cache = past_key_value[0]
            self.v_cache = past_key_value[1]
            self.past_key_values_length = self.token_offset[0]
            self.token_offset[:] = self.token_offset[0] + 1
        else:
            if not self.soc_info.need_nz:
                self.k_cache = [torch.zeros(self.batch_num,
                                            self.max_position_embeddings,
                                            self.num_key_value_heads * self.head_size, device=input_ids.device,
                                            dtype=torch.float16) for _ in range(self.num_layers)]
                self.v_cache = [torch.zeros(self.batch_num,
                                            self.max_position_embeddings,
                                            self.num_key_value_heads * self.head_size, device=input_ids.device,
                                            dtype=torch.float16) for _ in range(self.num_layers)]
            else:
                self.k_cache = [torch_npu.npu_format_cast_(torch.zeros(self.batch_num,
                                math.ceil(self.num_key_value_heads * self.head_size / self.nz_dim),
                                self.max_position_embeddings, self.nz_dim, device=input_ids.device,
                                dtype=torch.float16), 29) for _ in range(self.num_layers)]
                torch.npu.empty_cache()
                self.v_cache = [torch_npu.npu_format_cast_(torch.zeros(self.batch_num,
                                math.ceil(self.num_key_value_heads * self.head_size / self.nz_dim),
                                self.max_position_embeddings, self.nz_dim, device=input_ids.device,
                                dtype=torch.float16), 29) for _ in range(self.num_layers)]
                torch.npu.empty_cache()
            self.past_key_values_length = 0
            self.token_offset[:] = input_ids.shape[1]
            self.seq_len_encoder[:] = input_ids.shape[1]

    def init_position_ids(self, input_ids, position_ids):

        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(
                self.past_key_values_length, seq_length + self.past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        return position_ids

    def init_mask(self, input_ids, attention_mask):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=device
            )
        combined_attention_mask = None
        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_ids.shape,
                torch.float16,
                device=device,
                past_key_values_length=self.past_key_values_length,
            )
        attention_mask = _expand_mask(attention_mask, torch.float16, tgt_len=seq_length).to(device)
        attention_mask = attention_mask if combined_attention_mask is None else attention_mask + combined_attention_mask
        dim_0 = attention_mask.shape[2]
        dim_1 = attention_mask.shape[3]
        if not self.soc_info.need_nz:
            self.mask_full[:batch_size, :dim_0, :dim_1] = attention_mask.squeeze(1)
        else:
            self.mask_full = torch.zeros((self.batch_num, self.max_position_embeddings, 
                self.max_position_embeddings), dtype=torch.half, device=input_ids.device)
            self.mask_full[:batch_size, :dim_0, :dim_1] = attention_mask.squeeze(1)
            self.mask_full = torch_npu.npu_format_cast_(
                self.mask_full.view(self.batch_num, self.mask_full.shape[1],
                self.mask_full.shape[2] // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)

    def prepare_inputs_for_ascend(self,
                                  input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  cu_seqlen_prefill: Optional[bool],
                                  max_seq_len: int,
                                  ):
        if self.num_attention_heads == self.num_key_value_heads:
            cos_embed, sin_embed = self.ascend_rotary_embedding.get_cos_sin_total(
                position_ids, max_seq_len, torch.float16
            )
        else:
            self.ascend_rotary_embedding.update_cos_sin_cache_total(torch.float16, position_ids.device, max_seq_len)
            cos_embed = self.ascend_rotary_embedding.get_cos_cached_total()
            sin_embed = self.ascend_rotary_embedding.get_sin_cached_total()

        if self.soc_info.need_nz:
            pass

        if cu_seqlen_prefill:
            acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": [input_ids.shape[1]] * self.batch_num
            })
            acl_encoder_operation_inputs = [input_ids, position_ids, cos_embed, sin_embed, self.mask_full]
            acl_encoder_operation_inputs.extend(self.k_cache)
            acl_encoder_operation_inputs.extend(self.v_cache)
            acl_encoder_operation_inputs.append(self.token_offset)
            acl_encoder_operation_inputs.append(self.seq_len_encoder)
            acl_encoder_operation_inputs.extend(self.layer_ids)
            return acl_encoder_operation_inputs, acl_param
        else:
            acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": [1] * self.batch_num
            })
            acl_decoder_operation_inputs = [input_ids, position_ids, cos_embed, sin_embed, self.mask_full]
            acl_decoder_operation_inputs.extend(self.k_cache)
            acl_decoder_operation_inputs.extend(self.v_cache)
            acl_decoder_operation_inputs.append(self.token_offset)
            acl_decoder_operation_inputs.append(self.seq_len_decoder)
            acl_decoder_operation_inputs.extend(self.layer_ids)
            return acl_decoder_operation_inputs, acl_param

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                cu_seqlen_prefill):
        if cu_seqlen_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    @abstractmethod
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_kvcache(input_ids, past_key_values)
        position_ids = self.init_position_ids(input_ids, position_ids)
        self.init_mask(input_ids, attention_mask)

        cu_seqlen_prefill = True if not past_key_values else False
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            self.max_position_embeddings,
        )
        logits = self.execute_ascend_operator(acl_inputs, acl_param, cu_seqlen_prefill)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        next_cache = [self.k_cache, self.v_cache] if use_cache else None
        if not return_dict:
            return (loss,) + tuple(v for v in [logits, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

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

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head.linear

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.linear = new_embeddings
