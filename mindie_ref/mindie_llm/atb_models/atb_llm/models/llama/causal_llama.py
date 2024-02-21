# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from typing import Optional
import torch

from atb_llm.utils.layers import load_column_multi
from ..base.causal_lm import CausalLM
from .modeling_llama import LlamaModel, LlamaConfig


class LlamaForCausalLM(CausalLM):
    def __init__(self, config, weights):
        super().__init__(config, weights)
        self.model = LlamaModel(config, weights)

        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.placeholder = torch.zeros(1, dtype=torch.float16).npu()
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32).npu()
        self.in_beta = torch.zeros(config.hidden_size, dtype=torch.float16).npu()
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()

    def init_ascend_operations(self, config: LlamaConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama_parallel_decoder_model")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama_parallel_decoder_model")

        # 设置模型参数
        coder_param = {
            "isFA": True,
            "isBF16": False,
            "isPack": True,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "quantType": 2 if self.quantize == "smooth_quant" else 0,
            "rmsNormEps": config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "numHiddenLayers": self.num_layers,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "tokenOffset": [0],
            "seqLen": [1],
        }
        encoder_param = {**coder_param, "isPrefill": True}
        decoder_param = {**coder_param, "isPrefill": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

    def init_ascend_weight(self):
        weights = [self.model.state_dict()["embed_tokens.weight"]]
        attn_layer_names = [
            'self_attn.q_proj.linear', 'self_attn.k_proj.linear',
            'self_attn.v_proj.linear', 'self_attn.o_proj.linear'
        ]
        mlp_layer_names = [
            'mlp.gate_proj.linear', 'mlp.up_proj.linear', 'mlp.down_proj.linear'
        ]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.model.layers[i].state_dict()
            if self.quantize == "smooth_quant":
                weights_t.append(weights_layer["input_layernorm.weight"])
                for layer_name in attn_layer_names:
                    weights_t.append(weights_layer[f'{layer_name}.weight'])
                    weights_t.append(weights_layer[f'{layer_name}.act_scales'])
                    weights_t.append(weights_layer[f'{layer_name}.act_zeros'])
                    weights_t.append(weights_layer[f'{layer_name}.output_scales'])
                weights_t.append(weights_layer["post_attention_layernorm.weight"])
                for layer_name in mlp_layer_names:
                    weights_t.append(weights_layer[f'{layer_name}.weight'])
                    weights_t.append(weights_layer[f'{layer_name}.act_scales'])
                    weights_t.append(weights_layer[f'{layer_name}.act_zeros'])
                    weights_t.append(weights_layer[f'{layer_name}.output_scales'])
            else:
                weights_t.append(weights_layer["input_layernorm.weight"])
                weights_t.append(self.weight_format_cast(weights_layer["self_attn.query_key_value.linear.weight"]))
                weights_t.extend([self.placeholder] * 11)
                weights_t.append(self.weight_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))
                weights_t.extend([self.placeholder] * 3)
                weights_t.append(weights_layer["post_attention_layernorm.weight"])
                weights_t.append(self.weight_format_cast(weights_layer["mlp.gate_up_proj.linear.weight"]))
                weights_t.extend([self.placeholder] * 7)
                weights_t.append(self.weight_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
                weights_t.extend([self.placeholder] * 3)
            if self.soc_info.need_nz:
                del self.model.layers[i].self_attn
                del self.model.layers[i].post_attention_layernorm
                del self.model.layers[i].mlp
            weights.extend(weights_t)

        weights.append(self.model.state_dict()["norm.weight"])
        weights.append(self.weight_format_cast(self.state_dict()["lm_head.linear.weight"]))

        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)

    def init_kvcache(self, input_ids, past_key_value):
        super().init_kvcache(input_ids, past_key_value)
        self.acl_encoder_operation.set_kv_cache(self.k_cache, self.v_cache)
        self.acl_decoder_operation.set_kv_cache(self.k_cache, self.v_cache)

    def prepare_inputs_for_ascend(self,
                                  input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  cu_seqlen_prefill: Optional[bool],
                                  max_seq_len: int,
                                  ):
        self.ascend_rotary_embedding.update_cos_sin_cache_total(torch.float16, position_ids.device, max_seq_len)
        cos_embed = self.ascend_rotary_embedding.get_cos_cached_total()
        sin_embed = self.ascend_rotary_embedding.get_sin_cached_total()

        acl_operation_inputs = [
            input_ids, position_ids, cos_embed, sin_embed, self.mask_full,
            self.placeholder, self.placeholder, self.kv_cache_idx, self.token_offset,
            self.placeholder, self.in_beta,
            self.seq_len_encoder if cu_seqlen_prefill else self.seq_len_decoder,
            torch.tensor([self.seq_len_encoder[0] - 1], dtype=torch.int64,
                         device="npu") if cu_seqlen_prefill else self.lm_head_indices_fake
        ]
        acl_param = json.dumps({
            "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
            "seqLen": [input_ids.shape[1]] * self.batch_num if cu_seqlen_prefill else self.acl_param_seq_len_decoder
        })

        return acl_operation_inputs, acl_param
