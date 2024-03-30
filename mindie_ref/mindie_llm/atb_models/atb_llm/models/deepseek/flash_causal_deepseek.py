# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os
import math
from typing import Optional, List, Tuple

import torch

from ..base.flash_causal_lm import FlashForCausalLM
from .modeling_deepseek import DeepseekConfig, FlashDeepseekModel
from ...utils.data.weight_wrapper import AttnModuleNames, MlpModuleNames, WeightWrapper
from atb_llm.utils.layers import (
    TensorEmbedding,
    load_column_multi,
)


class FlashDeepseekForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights):
        # called the:
        # self.init_ascend_operations
        super().__init__(config, weights)
        self.model = FlashDeepseekModel(config, weights)
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
        self.tp = config.tp if config.tp else False
        if self.tp:
            self.expert_parallel_degree = 1
            self.maskStartIdx = 0
        else:
            self.expert_parallel_degree = self.tp_world_size
            self.maskStartIdx = self.tp_rank

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

    def get_weights(self):
        attn_module_names = AttnModuleNames(
            norm_name='input_layernorm',
            pack_name='self_attn.query_key_value',
            q_name='self_attn.q_proj',
            k_name='self_attn.k_proj',
            v_name='self_attn.v_proj',
            o_name='self_attn.o_proj'
        )
        mlp_module_names = MlpModuleNames(
            norm_name='post_attention_layernorm',
            pack_name='mlp.gate_up_proj',
            gate_name='mlp.gate_proj',
            up_name='mlp.up_proj',
            down_name='mlp.down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_module_names, mlp_module_names)
        weight_wrapper.register_embedding(self.model.state_dict(), 'embed_tokens')
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            layer_dict = layer.state_dict()
            weight_wrapper.layer_linear_type.clear()
            # add input layernorm and self attn weights
            weight_wrapper.register_layer_attn(layer_dict, layer.self_attn.pack_type, self.quantize, attn_module_names)
            # add post norm weights
            weight_wrapper.weights.append(layer_dict["post_attention_layernorm.weight"])

            if i == 0:
                # add shared experts weights
                mlp_layer_names = ['mlp.gate_up_proj.linear', 'mlp.down_proj.linear']
                for layer_name in mlp_layer_names:
                    weight_wrapper.weights.append(layer_dict[f'{layer_name}.weight'])
                # add gate weights
                weight_wrapper.weights.append(self.placeholder)

                # add common experts
                COMMON_EXPERTS_NUM = 64
                weight_wrapper.weights.extend([self.placeholder] * 2 * COMMON_EXPERTS_NUM)
            else:
                # add shared experts weights
                shared_experts_layer_names = ['mlp.shared_experts.gate_up_proj.linear', 'mlp.shared_experts.down_proj.linear']
                for layer_name in shared_experts_layer_names:
                    weight_wrapper.weights.append(layer_dict[f'{layer_name}.weight'])

                # add gate weights
                weight_wrapper.weights.append(layer_dict["mlp.gate.weight"])

                # add common experts
                COMMON_EXPERTS_NUM = 64
                if self.tp:
                    for j in range(COMMON_EXPERTS_NUM):
                        weight_wrapper.weights.append(layer_dict[f"mlp.experts.{j}.gate_up_proj.linear.weight"])
                        weight_wrapper.weights.append(layer_dict[f"mlp.experts.{j}.down_proj.linear.weight"])
                else:
                    if self.expert_parallel_degree == 0:
                        raise ValueError(
                            f"ERROR: Expert parallel degree is zero which is invalid "
                        )
                    else:
                        expert_per_rank = COMMON_EXPERTS_NUM / self.expert_parallel_degree
                    for j in range(COMMON_EXPERTS_NUM):
                        if j < expert_per_rank:
                            exprt_id = int(j + self.tp_rank * expert_per_rank)                            
                            weight_wrapper.weights.append(torch.cat([layer_dict[f"mlp.experts.{j}.expert_gate_proj"],
                                                        layer_dict[f"mlp.experts.{j}.expert_up_proj"]]))
                            weight_wrapper.weights.append(layer_dict[f"mlp.experts.{j}.expert_down_proj"])
                        else:   
                            weight_wrapper.weights.extend([self.placeholder] * 2)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
        weight_wrapper.register_model_norm(self.model.state_dict(), 'norm')
        weight_wrapper.register_model_lmhead(self.state_dict(), 'lm_head')
        return weight_wrapper.weights

    def init_ascend_weight(self):
        # add embedding
        self.ascend_weight = self.get_weights()

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
            "expertParallelDegree": self.expert_parallel_degree,
            "maskStartIdx": self.maskStartIdx,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz or str(os.getenv("RANKTABLEFILE", "")) else "lccl",
            "rankTableFile": str(os.getenv("RANKTABLEFILE", ""))
        }

        encoder_param = {**coder_param, "isPrefill": True, "supportLcoc": False if self.soc_info.need_nz else True}
        decoder_param = {**coder_param, "isPrefill": False, "supportLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        # self.init_params()
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

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