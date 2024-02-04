# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
import math
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
import torch_npu
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

from atb_llm.utils.log import logger
from atb_llm.models.baichuan.v2_13b.config import BaichuanConfig
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    TensorParallelHead,
    AttentionMask
)


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
    def __init__(self, prefix, weights, epsilon=1e-6):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.epsilon = epsilon


class BaichuanAttention(torch.nn.Module):
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

        self.softmax_scale = self.head_size ** -0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.query_key_value = TensorParallelColumnLinear.load_qkv(
            config, prefix=f"{prefix}.W_pack", weights=weights, bias=False
        )

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

        self.prefix = prefix


class MLP(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
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


class BaichuanLayer(torch.nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"

        self.self_attn = BaichuanAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.input_layernorm = RMSNorm(prefix=f"{prefix}.input_layernorm", weights=weights, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(prefix=f"{prefix}.post_attention_layernorm", weights=weights,
                                                epsilon=config.rms_norm_eps)


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


def ntokens_trans_data_attention_mask(tensor, is_prefill=False):
    """
    prefill: [batch , head_num,max_s,max_s] -> [batch * head_num, maxS/16, maxS, 16]
    prefill: [4, 40, 1024, 1024]  ->  [160, 64, 1024, 16]
    max_s不够16整除的要pad 如[4,40,17,17] -> [4, 40, 17, 32] -> [160,2,17,16]

    decode: [batch , head_num,1,max_s] -> [batch * head_num, max_s/16, 16, 16]
    max_s不够16整除的要pad 如[1,40,1,17] -> [1, 40, 1, 32] -> [1, 40, 16, 32] ->[40,2,16,16]
    """
    logger.debug(f"shape of tensor in {is_prefill=} before transdata  is {tensor.shape}")
    nz_dim = 16
    if is_prefill:
        return torch_npu.npu_format_cast(tensor.view(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3] // nz_dim,
            nz_dim
        ).transpose(1, 2).contiguous(), 29)
    else:
        tensor = tensor.repeat(1, 1, nz_dim, 1)
        return torch_npu.npu_format_cast(tensor.view(
            tensor.shape[0] * tensor.shape[1],
            nz_dim,
            tensor.shape[3] // nz_dim,
            nz_dim
        ).transpose(1, 2).contiguous(), 29)


class BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = TensorEmbedding(prefix="model.embed_tokens", weights=weights)
        self.layers = nn.ModuleList(
            [BaichuanLayer(layer_id, config, weights, ) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(prefix="model.norm", weights=weights, epsilon=config.rms_norm_eps)

        self.max_cache_pos = config.model_max_length
        self.first_run = True
        self.alibi_mask = None
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
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None

        self.ascend_atten_mask = AttentionMask.static(config.model_max_length)
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')

        self.batch_size = 0

    def maybe_format_cast(self, tensor):
        """
        maybe_format_cast
        """
        if not self.soc_info.need_nz:  # transdata 会额外占资源
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def init_ascend_operations(self, config: BaichuanConfig):
        self.acl_param_encoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": config.num_attention_heads // self.tp_world_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": True,
            "backend": os.getenv("BACKEND", "hccl"),
            "isLmHeadParallel": not self.soc_info.need_nz
        })
        self.acl_param_decoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": config.num_attention_heads // self.tp_world_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": False,
            "backend": os.getenv("BACKEND", "hccl"),
            "isLmHeadParallel": not self.soc_info.need_nz
        })
        logger.info(self.acl_param_encoder)
        logger.info(self.acl_param_decoder)
        logger.info("using flash_baichuan2_13b_modeling_ascend")

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_pa_model")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_13b_pa_model")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.num_layers = config.num_hidden_layers

        self.acl_operation_inputs = []
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")
        self.lm_head_weight = None
        self.is_prefill = True

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
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.query_key_value.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))
            weights_t.append(weights_layer["post_attention_layernorm.weight"])
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.gate_up_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
            weights.extend(weights_t)
            if self.soc_info.need_nz:
                del self.layers[i].self_attn
                del self.layers[i].mlp
            torch.npu.synchronize()
            peak_memory = torch_npu.npu.max_memory_allocated()
            logger.warning(f">>>>layer {i} peak_memory {peak_memory / 1024 / 1024} MB")
            torch.npu.synchronize()
        weights.append(self.state_dict()["norm.weight"])
        weights.append(self.lm_head_weight)
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
            self.acl_param_encoder = json.dumps({
                "seqLen": input_lengths.tolist()
            })

        self.acl_operation_inputs = [
            input_ids,
            attention_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            input_lengths.to(torch.int32),
            lm_head_indices if self.is_prefill else self.lm_head_indices_fake,
            self.place_holder
        ]

        return self.acl_operation_inputs, self.acl_param_encoder if self.is_prefill else self.acl_param_decoder

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
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(
            input_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            lm_head_indices,
            attention_mask)
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

    def generate_mask(self, max_s, kv_cache):
        """
        生成mask
        """
        pad_max_s = max_s
        if self.soc_info.need_nz:
            nz_pad = math.ceil(max_s / 16) * 16 - max_s
            pad_max_s = max_s + nz_pad
        attention_mask = self.ascend_atten_mask.get_attn_mask(pad_max_s, kv_cache[0][0].dtype, kv_cache[0][0].device)

        total_alibi_mask = self.get_alibi_mask(self.place_holder, pad_max_s)
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

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_s: int,  # 最长的request长度
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
        logger.debug(f"{max_s=}")
        """
        self.is_prefill=True
        input_ids.shape=torch.Size([4096])
        block_tables=tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
        [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23, 24, 25, 26],
        [27, 28, 29, 30, 31, 32, 33, 34, 35]], device='npu:0', dtype=torch.int32)
        block_tables.shape=torch.Size([4, 9])
        slots=tensor([   0,    1,    2,  ..., 4477, 4478, 4479], device='npu:0',dtype=torch.int32)
        slots.shape=torch.Size([4096])
        input_lengths=tensor([1024, 1024, 1024, 1024], device='npu:0')
        input_lengths.shape=torch.Size([4])
        max_s=1024
        """

        """
        input_ids.shape=torch.Size([1])
        block_tables=tensor([[0]], device='npu:0', dtype=torch.int32)
        block_tables.shape=torch.Size([1, 1])
        slots=tensor([16], device='npu:0', dtype=torch.int32)
        slots.shape=torch.Size([1])
        input_lengths=tensor([17], device='npu:0')
        input_lengths.shape=torch.Size([1])
        max_s=17
        self.n_head=40
        attention_mask.shape=torch.Size([17, 17])
        total_alibi_mask.shape=torch.Size([1, 40, 17, 17])
        final attention_mask shape in self.is_prefill=False is torch.Size([1, 40, 1, 17])
        shape of tensor in is_prefill=False before transdata  is torch.Size([1, 40, 1, 17])

        """
        attention_mask = self.generate_mask(max_s, kv_cache)
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
            max_s,
            lm_head_indices,
            attention_mask)

        return tuple(v for v in [hidden_states] if v is not None)


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


class FlashBaichuanForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        load_atb_speed()
        self.model = BaichuanModel(config, weights)
        self.soc_info = NPUSocInfo()
        logger.info(self.soc_info)
        # Initialize weights and apply final processing
        self.lm_head_weight = None
        self.parallel_lm_head = not self.soc_info.need_nz  # 310P 暂时不支持ALLGather算子
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
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.lm_head_weight is None:
            if self.config.vocab_size == 125696:
                logger.debug("baichuan2 13B normalize lm_head")
                self.lm_head_weight = nn.functional.normalize(self.state_dict()["lm_head.linear.weight"])
            else:
                self.lm_head_weight = self.state_dict()["lm_head.linear.weight"]
            if self.soc_info.need_nz:
                self.lm_head_weight.data = torch_npu.npu_format_cast(self.lm_head_weight.data, 29)
            self.model.lm_head_weight = self.lm_head_weight

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
