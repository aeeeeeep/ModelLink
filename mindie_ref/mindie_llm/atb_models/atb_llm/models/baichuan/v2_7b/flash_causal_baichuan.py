# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
import math
from typing import Optional, List, Tuple

import torch
import torch.distributed
import torch_npu
from atb_llm.utils.initial import NPUSocInfo, load_atb_speed
from atb_llm.utils.layers import (
    TensorParallelColumnLinear,
    TensorEmbedding,
    PositionRotaryEmbedding,
    AttentionMask,
    TensorParallelHead,
    load_column_multi,
    load_row
)
from atb_llm.utils.log import logger
from torch import nn
from transformers.activations import ACT2FN


class RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class FlashBaichuanAttention(torch.nn.Module):
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
        self.query_key_value = TensorParallelColumnLinear.load_qkv(
            config, prefix=f"{prefix}.W_pack", weights=weights, bias=False, head_size=self.head_size
        )

        self.o_proj = load_row(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            head_size=self.head_size
        )

        self.prefix = prefix


class MLP(nn.Module):
    def __init__(self, prefix, config, weights):
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
        # Fuse gate and up proj

        self.gate_up_proj = load_column_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            head_size=1
        )

        self.down_proj = load_row(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            head_size=1
        )

        try:
            self.intermediate_size = (math.ceil(config.intermediate_size / weights.process_group.size()))
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e


class FlashDecoderLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashBaichuanAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = RMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )


class FlashBaichuanModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashDecoderLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False


class FlashBaichuanForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        load_atb_speed()
        self.config = config
        self.soc_info = NPUSocInfo()
        self.model = FlashBaichuanModel(config, weights)
        self.parallel_lm_head = True

        if self.parallel_lm_head:
            self.lm_head = load_column_multi(
                config,
                prefixes=["lm_head"],
                weights=weights,
                head_size=1,
                lm_head=True,
                norm=self.config.vocab_size == 125696
            )
        else:
            self.lm_head = TensorParallelHead.load_weight(
                config,
                prefix="lm_head",
                weights=weights,
                is_norm=True  # 不生效的配置
            )

        # for ascend init

        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        try:
            self.num_heads = math.ceil(self.num_heads / weights.process_group.size())
            self.num_key_value_heads = math.ceil(config.num_key_value_heads / weights.process_group.size())
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
        self.lm_head_weight = None
        self.is_prefill = True
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

    def init_ascend_operations(self, config):
        self.acl_param_encoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": self.num_heads,
            "dk": self.head_size,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": True,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "isLmHeadParallel": self.parallel_lm_head
        })
        self.acl_param_decoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": self.num_heads,
            "dk": self.head_size,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": False,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "isLmHeadParallel": self.parallel_lm_head
        })
        self.max_position_embeddings = config.max_position_embeddings
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_7b_PagedAttentionModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("baichuan2_7b_PagedAttentionModel")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.acl_operation_inputs = []
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.ascend_atten_mask = AttentionMask.static(config.max_position_embeddings)
        self.ascend_atten_mask_fake = self.ascend_atten_mask.get_attn_mask(1, dtype=torch.float16, device="npu")

    def init_ascend_weight(self):
        weights = [self.model.state_dict()["embed_tokens.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.model.layers[i].state_dict()
            weights_t.append(weights_layer["input_layernorm.weight"])
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.query_key_value.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))
            weights_t.append(weights_layer["post_attention_layernorm.weight"])
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.gate_up_proj.linear.weight"]))
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
            weights.extend(weights_t)
            if self.soc_info.need_nz:
                del self.model.layers[i].self_attn
                del self.model.layers[i].mlp
        weights.append(self.model.state_dict()["norm.weight"])
        self.lm_head_weight = self.state_dict()["lm_head.linear.weight"]
        weights.append(self.maybe_format_cast(self.lm_head_weight))

        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)

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
