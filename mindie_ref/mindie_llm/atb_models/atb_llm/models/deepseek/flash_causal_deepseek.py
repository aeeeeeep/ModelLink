# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os
import math
from typing import Optional, List, Tuple

import torch

from transformers.configuration_utils import PretrainedConfig

from ..base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorEmbedding,
    load_column_multi,
)


class DeepseekConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=102400,
            hidden_size=2048,
            intermediate_size=10944,
            moe_intermediate_size=1408,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=16,
            n_shared_experts=2,
            n_routed_experts=64,
            num_experts_per_tok=None,
            moe_layer_freq=1,
            first_k_dense_replace=0,
            norm_topk_prob=False,
            scoring_func='softmax',
            aux_loss_alpha=0.001,
            seq_aux=True,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=100000,
            eos_token_id=100001,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            world_size=8,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.world_size = world_size
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class FlashDeepseekForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights):
        # called the:
        # self.init_ascend_operations
        super().__init__(config, weights)
        self.weights = weights
        self.config = config
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.config = config
        self.in_tensor_length = 15
        self.acl_encoder_operation_inputs = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs = [None] * self.in_tensor_length

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.padding_idx = config.pad_token_id
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.hidden_dim = config.hidden_size
        self.final_hidden_states = []
        self.one_hot_one = torch.ones(1, dtype=torch.int32, device="npu")
        self.one_hot_zero = torch.zeros(1, dtype=torch.int32, device="npu")

    # called by super().prepare_inputs_for_ascend
    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: DeepseekConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("deepseekDense_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("deepseekDense_DecoderModel")


    def init_ascend_weight(self):
        # add embedding
        weights = [self.weights.get_tensor("model.embed_tokens.weight").npu()]

        IN_INPUT_NORM_PLACEHOLDER_NUM = 3
        IN_QKV_PLACEHOLDER_NUM = 14
        IN_ATTENTION_OUT_PLACEHOLDER = 4

        for i in range(self.num_layers):
            weights_t = []

            # add input layernorm weights
            weights_t.append(self.weights.get_tensor(f'model.layers.{i}.input_layernorm.weight').npu())
            for k in range(IN_INPUT_NORM_PLACEHOLDER_NUM):
                weights_t.append(self.placeholder)

            # add attention weights
            query_key_value = load_column_multi(
                self.config,
                prefixes=[f"model.layers.{i}.self_attn.q_proj",
                          f"model.layers.{i}.self_attn.k_proj",
                          f"model.layers.{i}.self_attn.v_proj"],
                weights=self.weights,
                head_size=self.config.hidden_size // self.config.num_attention_heads
            ).npu()
            weights_t.append(query_key_value.linear.weight)

            for k in range(IN_QKV_PLACEHOLDER_NUM):
                weights_t.append(self.placeholder)

            o_proj = TensorParallelRowLinear.load(
                self.config,
                prefix=f"model.layers.{i}.self_attn.o_proj",
                weights=self.weights,
                bias=False,
            ).npu()
            weights_t.append(o_proj.linear.weight)
            for k in range(IN_ATTENTION_OUT_PLACEHOLDER):
                weights_t.append(self.placeholder)

            # add post norm weights
            weights_t.append(self.weights.get_tensor(f'model.layers.{i}.post_attention_layernorm.weight').npu())

            if i == 0:
                # add shared experts weights
                gate_up = load_column_multi(
                    self.config,
                    prefixes=[f"model.layers.{i}.mlp.gate_proj",
                              f"model.layers.{i}.mlp.up_proj"],
                    weights=self.weights,
                    head_size=1
                ).npu()
                weights_t.append(gate_up.linear.weight)

                down_proj = TensorParallelRowLinear.load(
                    self.config,
                    prefix=f"model.layers.{i}.mlp.down_proj",
                    weights=self.weights,
                    bias=False,
                ).npu()
                weights_t.append(down_proj.linear.weight)
                # add gate weights
                weights_t.append(self.placeholder)

                # add common experts
                COMMON_EXPERTS_NUM = 64
                for j in range(COMMON_EXPERTS_NUM):
                    weights_t.append(self.placeholder)
                    weights_t.append(self.placeholder)
            else:
                # add shared experts weights
                shared_gate_up = load_column_multi(
                    self.config,
                    prefixes=[f"model.layers.{i}.mlp.shared_experts.gate_proj",
                              f"model.layers.{i}.mlp.shared_experts.up_proj"],
                    weights=self.weights,
                    head_size=1
                ).npu()
                weights_t.append(shared_gate_up.linear.weight)

                shared_down_proj = TensorParallelRowLinear.load(
                    self.config,
                    prefix=f"model.layers.{i}.mlp.shared_experts.down_proj",
                    weights=self.weights,
                    bias=False,
                ).npu()
                weights_t.append(shared_down_proj.linear.weight)
                # add gate weights
                weights_t.append(self.weights.get_tensor(f"model.layers.{i}.mlp.gate.weight").npu())

                # add common experts
                COMMON_EXPERTS_NUM = 64
                for j in range(COMMON_EXPERTS_NUM):
                    experts_gate_up = load_column_multi(
                        self.config,
                        prefixes=[f"model.layers.{i}.mlp.experts.{j}.gate_proj",
                                  f"model.layers.{i}.mlp.experts.{j}.up_proj"],
                        weights=self.weights,
                        head_size=1
                    ).npu()
                    weights_t.append(experts_gate_up.linear.weight)

                    experts_down_proj = TensorParallelRowLinear.load(
                        self.config,
                        prefix=f"model.layers.{i}.mlp.experts.{j}.down_proj",
                        weights=self.weights,
                        bias=False,
                    ).npu()
                    weights_t.append(experts_down_proj.linear.weight)

            # add layer weights
            weights.extend(weights_t)

        weights.append(self.weights.get_tensor("model.norm.weight").npu())
        weights.append(self.weights.get_tensor("lm_head.weight").npu())

        coder_param = {
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": False,
            "isBF16": False,
            "packQuantType": [[0, 0] for _ in range(self.config.num_hidden_layers)],
            "linearQuantType": [[0, 0, 0, 0, 0, 0, 0] for i in range(self.config.num_hidden_layers)],
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz or str(os.getenv("RANKTABLEFILE", "")) else "lccl",
            "RANKTABLEFILE": str(os.getenv("RANKTABLEFILE", ""))
        }

        encoder_param = {**coder_param, "isPrefill": True, "supportLcoc": False if self.soc_info.need_nz else True}
        decoder_param = {**coder_param, "isPrefill": False, "supportLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        # self.init_params()
        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)

    # called by super().forward()
    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                            self.device,
                                                            self.max_position_embeddings)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        hidden_states = self.embed_tokens(input_ids)
        input_length = hidden_states.shape[0]
        
        self.final_hidden_states = torch.zeros(
            (input_length, self.hidden_dim), dtype=hidden_states.dtype, device="npu"
        )

        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype,
                                                                    kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.attn_mask.get_attn_mask(self.max_position_embeddings, kv_cache[0][0].dtype,
                                                                    kv_cache[0][0].device)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                dtype=torch.int64, device=input_ids.device)
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
            self.acl_encoder_operation_inputs[7] = self.placeholder
            self.acl_encoder_operation_inputs[8] = self.placeholder
            self.acl_encoder_operation_inputs[9] = self.placeholder
            self.acl_encoder_operation_inputs[10] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[11] = lm_head_indices.to(torch.int64)
            self.acl_encoder_operation_inputs[12] = self.one_hot_one
            self.acl_encoder_operation_inputs[13] = self.one_hot_zero
            self.acl_encoder_operation_inputs[14] = self.final_hidden_states

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
                                                                    device=self.device)
            else:
                self.acl_decoder_operation_inputs[4] = self.attn_mask_fake
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = self.placeholder
            self.acl_decoder_operation_inputs[8] = self.placeholder
            self.acl_decoder_operation_inputs[9] = self.placeholder
            self.acl_decoder_operation_inputs[10] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[11] = self.lm_head_indices_fake
            self.acl_decoder_operation_inputs[12] = self.one_hot_one
            self.acl_decoder_operation_inputs[13] = self.one_hot_zero
            self.acl_decoder_operation_inputs[14] = self.final_hidden_states

            return self.acl_decoder_operation_inputs, self.acl_param

if __name__ == "__main__":
    config = DeepseekConfig()
    weights = None
    model = FlashDeepseekForCausalLM(config, weights)