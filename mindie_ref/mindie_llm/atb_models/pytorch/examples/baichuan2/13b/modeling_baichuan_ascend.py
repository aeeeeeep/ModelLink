import json
import math
import os
from contextlib import contextmanager
from threading import Thread
from typing import List, Optional, Tuple, Union

import torch
import torch_npu
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation.utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging, ContextManagers

from .configuration_baichuan import BaichuanConfig
from .generation_utils import build_chat_input, TextIterStreamer

logger = logging.get_logger(__name__)


def is_nd():
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version in [104, 220, 221, 222, 223, 224]


IS_ND = is_nd()


def get_rank_and_world_size():
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except:
        rank = 0
        world_size = 1
    return rank, world_size


RANK, WORLD_SIZE = get_rank_and_world_size()


def load_ascend_transformer():
    ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
    if ATB_SPEED_HOME_PATH is None:
        raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
    LIB_PATH = os.path.join(ATB_SPEED_HOME_PATH, "lib/libatb_speed_torch.so")
    torch.classes.load_library(LIB_PATH)


load_ascend_transformer()


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


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, epsilon=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        # convert into half-precision
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class MLP(torch.nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class BaichuanAttention(torch.nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.config = config
        self.world_size = WORLD_SIZE
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = int(os.getenv("MAX_SEQ_LEN", config.model_max_length))

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.num_heads = self.num_heads // self.world_size
        self.W_pack = torch.nn.Linear(self.hidden_size, 3 * self.num_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = (
            proj.unflatten(-1, (3, self.num_heads * self.head_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = (key_states, value_states) if use_cache else None
        # delete xops
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        # reduce
        if self.world_size >= 2:
            torch.distributed.all_reduce(attn_output, op=torch.distributed.ReduceOp.SUM)

        if not output_attentions:
            attn_weights = None
        past_key_value = (key_states, value_states) if use_cache else None

        return attn_output, attn_weights, past_key_value


class BaichuanLayer(torch.nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BaichuanAttention(config=config)
        self.world_size = WORLD_SIZE
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size // self.world_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # reduce
        if self.world_size >= 2:
            torch.distributed.all_reduce(
                hidden_states, op=torch.distributed.ReduceOp.SUM)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BaichuanPreTrainedModel(PreTrainedModel):
    config_class = BaichuanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BaichuanLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BaichuanModel):
            module.gradient_checkpointing = value


class KVAttentionManager:
    def __init__(self, config: BaichuanConfig, batch_size):
        self.nz_dim = 16
        self.is_full = True
        self.batch_size = batch_size
        self.num_layers = config.num_hidden_layers
        self.num_head = config.num_attention_heads // WORLD_SIZE
        self.hidden_size = config.hidden_size // WORLD_SIZE
        self.max_seq_len = int(os.getenv("MAX_SEQ_LEN", config.model_max_length))
        if not IS_ND:
            self.k_cache_input = torch.zeros(self.num_layers,
                                             self.batch_size,
                                             self.hidden_size // self.nz_dim,
                                             self.max_seq_len,
                                             self.nz_dim,
                                             device="npu", dtype=torch.half)
            self.v_cache_input = torch.zeros(self.num_layers,
                                             self.batch_size,
                                             self.hidden_size // self.nz_dim,
                                             self.max_seq_len,
                                             self.nz_dim,
                                             device="npu", dtype=torch.half)
            self.k_cache_input = torch_npu.npu_format_cast(self.k_cache_input, 29)
            torch.npu.empty_cache()
            self.v_cache_input = torch_npu.npu_format_cast(self.v_cache_input, 29)
        else:
            self.k_cache_input = torch.zeros(self.num_layers,
                                             batch_size,
                                             self.max_seq_len,
                                             self.hidden_size,
                                             device="npu", dtype=torch.half)
            self.v_cache_input = torch.zeros(self.num_layers,
                                             batch_size,
                                             self.max_seq_len,
                                             self.hidden_size,
                                             device="npu", dtype=torch.half)
        torch.npu.empty_cache()
        self.token_offset = 1
        self.attention_mask_max = torch.zeros(
            (self.batch_size, self.num_head, self.max_seq_len, self.max_seq_len),
            dtype=torch.half, device="npu")
        self.attention_mask_max_inc = torch.zeros(
            (self.batch_size, self.max_seq_len, self.max_seq_len), dtype=torch.half, device="npu")

    def init_attention_mask(self):
        self.attention_mask_max.zero_()
        self.attention_mask_max_inc.zero_()

    def init_seq_len_and_token_offset(self, seq_len):
        self.token_offset = seq_len
        self.seq_len_list_full = [self.token_offset] * self.batch_size
        self.seq_len_tensor_full = torch.full((self.batch_size,), self.token_offset, dtype=torch.int32).npu()
        self.seq_len_list_inc = [1] * self.batch_size
        self.seq_len_tensor_inc = torch.full((self.batch_size,), 1, dtype=torch.int32).npu()
        self.token_offset_tensor = torch.full((self.batch_size,), self.token_offset, dtype=torch.int32).npu()

    @property
    def seq_len_list(self):
        if self.is_full:
            return self.seq_len_list_full
        return self.seq_len_list_inc

    @property
    def seq_len_tensor(self):
        if self.is_full:
            return self.seq_len_tensor_full
        return self.seq_len_tensor_inc

    @property
    def token_offset_list(self):
        return [self.token_offset] * self.batch_size

    def trans_data(self, tensor):
        if self.is_full:
            return torch_npu.npu_format_cast(tensor.view(
                self.batch_size * self.num_head, self.max_seq_len,
                self.max_seq_len // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)
        else:

            return torch_npu.npu_format_cast(tensor.view(
                self.batch_size * self.num_head, self.nz_dim,
                self.max_seq_len // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)

    def get_attention_mask(self, attention_mask=None):
        if self.is_full:
            self.attention_mask_max[:, :, :self.token_offset, :self.token_offset] = attention_mask
            return self.trans_data(self.attention_mask_max) if not IS_ND else self.attention_mask_max
        else:
            if not IS_ND:
                self.attention_mask_max[:, :, :self.nz_dim, :self.token_offset] = attention_mask[:, :, -1:, :]
                return self.trans_data(self.attention_mask_max[:, :, :self.nz_dim, :])
            else:
                self.attention_mask_max[:, :, 0:1, :self.token_offset] = attention_mask[:, :, -1:, :]
                return self.attention_mask_max


class BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.rank = RANK
        self.world_size = WORLD_SIZE
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList([BaichuanLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()
        self.max_cache_pos = int(os.getenv("MAX_SEQ_LEN", config.model_max_length))
        self.first_run = True
        self.alibi_mask = None

        # for ascend init
        self.init_ascend_operations(config)
        self.layer_id_list = [torch.tensor([i], dtype=torch.int32).npu() for i in range(config.num_hidden_layers)]
        self.place_holder = torch.ones(1, dtype=torch.float16).npu()

    def init_ascend_operations(self, config: BaichuanConfig):
        head_size = config.hidden_size // config.num_attention_heads
        self.acl_param = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": config.num_attention_heads // self.world_size,
            "dk": head_size,
            "layerNum": config.num_hidden_layers,
            "rank": self.rank,
            "rankSize": self.world_size,
            "backend": os.getenv("BACKEND", "hccl")
        })
        self.max_position_embeddings = int(os.getenv("MAX_SEQ_LEN", config.model_max_length))
        self.acl_fa_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_flash_attention_model")

        self.acl_fa_operation.set_param(self.acl_param)

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.ascend_weight = []
        self.lm_head_weight = None
        self.batch_size = 0
        self.min_cache = torch.full(
            (self.max_position_embeddings, self.max_position_embeddings),
            torch.finfo(torch.half).min, dtype=torch.half).npu()

    def init_ascend_weight(self):
        weights = [self.state_dict()["embed_tokens.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.layers[i].state_dict()
            weights_t.append(weights_layer["input_layernorm.weight"])
            weights_t.append(weights_layer["self_attn.W_pack.weight"])
            weights_t.append(weights_layer["self_attn.o_proj.weight"])
            weights_t.append(weights_layer["post_attention_layernorm.weight"])
            weights_t.append(weights_layer["mlp.gate_proj.weight"])
            weights_t.append(weights_layer["mlp.down_proj.weight"])
            weights_t.append(weights_layer["mlp.up_proj.weight"])

            weights.extend(weights_t)
        weights.append(self.state_dict()["norm.weight"])
        weights.append(self.lm_head_weight)
        self.ascend_weight = weights
        self.acl_fa_operation.set_weight(self.ascend_weight)

    def prepare_inputs_for_ascend(self, input_ids, attention_mask=None, past_key_values=None):
        self.kv_attention_manager.is_full = not past_key_values
        seqlen_max = torch.tensor([self.kv_attention_manager.seq_len_tensor[0] - 1], dtype=torch.int64, device="npu")
        inputs = [input_ids,
                  self.kv_attention_manager.get_attention_mask(attention_mask),
                  self.kv_attention_manager.k_cache_input,
                  self.kv_attention_manager.v_cache_input,
                  self.kv_attention_manager.token_offset_tensor,
                  self.kv_attention_manager.seq_len_tensor,
                  self.place_holder,
                  seqlen_max,
                  ] + self.layer_id_list

        return inputs

    def execute_ascend_operator(self, input_ids, attention_mask=None, past_key_values=None):
        acl_inputs = self.prepare_inputs_for_ascend(input_ids, attention_mask, past_key_values)
        tmp_param = json.dumps(
            {"tokenOffset": self.kv_attention_manager.token_offset_list,
             "seqLen": self.kv_attention_manager.seq_len_list
             })
        acl_model_out = self.acl_fa_operation.execute(acl_inputs, tmp_param)
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
                self.register_buffer("future_mask", _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(tensor),
                                     persistent=False)
            if seq_length_with_past > self.max_cache_pos:
                self.max_cache_pos = seq_length_with_past
                self.register_buffer("future_mask", _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(tensor),
                                     persistent=False)
            mask = self.future_mask[:self.n_head, :seq_length_with_past, :seq_length_with_past]
            if self.world_size > 1:
                mask = mask.chunk(self.world_size, dim=0)
        return mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds simultaneously")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_length_with_past = seq_length

        # flash attention init
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.kv_attention_manager = KVAttentionManager(self.config, batch_size)

        if past_key_values is None:
            self.kv_attention_manager.init_seq_len_and_token_offset(seq_length)
            self.kv_attention_manager.init_attention_mask()

        if past_key_values is not None:
            past_key_values_length = self.kv_attention_manager.token_offset
            seq_length_with_past = seq_length_with_past + past_key_values_length
            self.kv_attention_manager.token_offset = self.kv_attention_manager.token_offset + 1
            self.kv_attention_manager.token_offset_tensor += 1
        inputs_embeds = self.place_holder
        alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)
        if self.world_size > 1:
            alibi_mask = alibi_mask[self.rank]

        if attention_mask is not None:
            if not past_key_values:
                self.ori_len_list = attention_mask.sum(dim=-1)
                if len(attention_mask.shape) == 2:
                    expanded_mask = attention_mask.to(alibi_mask.dtype)
                    expanded_mask = torch.tril(
                        torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                    ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
                else:
                    expanded_mask = attention_mask
                bsz = self.batch_size
                src_len, tgt_len = alibi_mask.size()[-2:]
                expanded_mask = expanded_mask.unsqueeze(1).expand(bsz, 1, src_len, tgt_len).to(alibi_mask.dtype)
                inverted_mask = 1.0 - expanded_mask
                inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                                          torch.finfo(alibi_mask.dtype).min)
                attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
                for i in range(self.batch_size):
                    ori_len = self.ori_len_list[i].item()
                    self.kv_attention_manager.attention_mask_max_inc[i][
                    :,
                    :self.kv_attention_manager.token_offset - ori_len
                    ] = self.min_cache[:, :self.kv_attention_manager.token_offset - ori_len]
            else:
                attention_mask = self.kv_attention_manager.attention_mask_max_inc[
                                 :,
                                 :self.kv_attention_manager.token_offset,
                                 :self.kv_attention_manager.token_offset].unsqueeze(1) + alibi_mask.unsqueeze(0)
        else:
            attention_mask = alibi_mask

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if not self.ascend_weight:
            self.init_ascend_weight()
        hidden_states = self.execute_ascend_operator(input_ids, attention_mask, past_key_values)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = (
            (self.kv_attention_manager.k_cache_input, self.kv_attention_manager.v_cache_input),) if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
            self.first_flag = True
        elif self.first_flag:
            self.first_flag = False
            self.weight.data = nn.functional.normalize(self.weight)
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


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


class BaichuanForCausalLM(BaichuanPreTrainedModel):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.model = BaichuanModel(config)
        self.lm_head = NormHead(config.hidden_size, config.vocab_size // WORLD_SIZE, bias=False)
        if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
            try:
                from .quantizer import quantize_offline, init_model_weight_int4
            except ImportError:
                raise ImportError(f"Needs quantize_offline to run quantize.")
            quantize_offline(self, 4)
        self.post_init()

        self.lm_head_weight = None

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

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            use_safetensors: bool = None,
            **kwargs,
    ):

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=False,
                proxies=None,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder="",
                _from_auto=False,
                _from_pipeline=None,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
            try:
                from .quantizer import init_model_weight_int4
                from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
                from accelerate.utils import CustomDtype
                from accelerate.utils import get_balanced_memory
            except ImportError:
                raise ImportError(f"Needs import model weight init func to run quantize.")
                # Instantiate model.
            init_contexts = [no_init_weights(_enable=True)]
            init_contexts.append(init_empty_weights())
            with ContextManagers(init_contexts):
                model = cls(config)

            model_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
            state_dict = torch.load(model_file, map_location="cpu")
            model.is_quantized = True

            device_map = kwargs.pop("device_map", None)
            torch_dtype = kwargs.pop("torch_dtype", None)
            if device_map is not None:
                kwargs = {"no_split_module_classes": model._no_split_modules}
                target_dtype = CustomDtype.INT4
                max_memory = get_balanced_memory(
                    model,
                    dtype=target_dtype,
                    low_zero=(device_map == "balanced_low_0"),
                    max_memory=None,
                    **kwargs,
                )
                kwargs["max_memory"] = max_memory
                device_map = infer_auto_device_map(model, dtype=target_dtype, **kwargs)
            model = init_model_weight_int4(config, model, state_dict)

            # Set model in evaluation mode to deactivate DropOut modules by default
            model.eval()
            # If it is a model with generation capabilities, attempt to load the generation config
            if model.can_generate():
                try:
                    model.generation_config = GenerationConfig.from_pretrained(
                        pretrained_model_name_or_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=False,
                        proxies=None,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder="",
                        _from_auto=False,
                        _from_pipeline=None,
                        **kwargs,
                    )
                except (OSError, TypeError):
                    logger.info(
                        "Generation config file not found, using a generation config created from the model config."
                    )
                    pass

            if device_map is not None:
                dispatch_model(model, device_map=device_map)

            return model

        return super(BaichuanForCausalLM, cls).from_pretrained(pretrained_model_name_or_path, *model_args,
                                                               config=config, cache_dir=cache_dir,
                                                               ignore_mismatched_sizes=ignore_mismatched_sizes,
                                                               force_download=force_download,
                                                               local_files_only=local_files_only, token=token,
                                                               revision=revision,
                                                               use_safetensors=use_safetensors, **kwargs)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.lm_head_weight is None:
            self.lm_head_weight = nn.functional.normalize(self.state_dict()["lm_head.weight"])
            if not IS_ND:
                self.lm_head_weight.data = torch_npu.npu_format_cast(self.lm_head_weight.data, 29)
            self.model.lm_head_weight = self.lm_head_weight

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def quantize(self, bits: int):
        try:
            from .quantizer import quantize_online
        except ImportError:
            raise ImportError(f"Needs QLinear to run quantize.")
        return quantize_online(self, bits)

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def _build_chat_input(
            self, tokenizer, messages: List[dict], max_new_tokens: int = 0
    ):
        model_max_length = int(os.getenv("MAX_SEQ_LEN", self.config.model_max_length))
        max_new_tokens = max_new_tokens or self.generation_config.max_new_tokens
        max_input_tokens = model_max_length - max_new_tokens
        max_input_tokens = max(model_max_length // 2, max_input_tokens)
        total_input, round_input = [], []
        for i, message in enumerate(messages[::-1]):
            content_tokens = tokenizer.encode(message["content"])
            if message["role"] == "user":
                round_input = (
                        [self.generation_config.user_token_id]
                        + content_tokens
                        + round_input
                )
                if (
                        total_input
                        and len(total_input) + len(round_input) > max_input_tokens
                ):
                    break
                else:
                    total_input = round_input + total_input
                    if len(total_input) >= max_input_tokens:
                        break
                    else:
                        round_input = []
            elif message["role"] == "assistant":
                round_input = (
                        [self.generation_config.assistant_token_id]
                        + content_tokens
                        + [self.generation_config.eos_token_id]
                        + round_input
                )
            else:
                raise ValueError(f"message role not supported yet: {message['role']}")
        total_input = total_input[-max_input_tokens:]  # truncate left
        total_input.append(self.generation_config.assistant_token_id)
        total_input = torch.LongTensor([total_input]).to(self.device)
        return total_input

    def stream_generate_ascend(self, **kwargs, ):
        device = kwargs.pop("device")
        torch.npu.set_device(device)
        return self.generate(**kwargs)

    def chat(self, tokenizer, messages: List[dict], stream=False, device: Optional[torch.device] = None,
             generation_config: Optional[GenerationConfig] = None):
        generation_config = generation_config or self.generation_config
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)

        print("model device: ", self.model.device)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.stream_generate_ascend, kwargs=dict(
                inputs=input_ids, streamer=streamer,
                generation_config=generation_config,
                device=device,
            )).start()
            return streamer
        else:
            outputs = self.generate(input_ids, generation_config=generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response
