import json
import os
import math
from typing import Optional
import torch
from loguru import logger
from ../..base.causal_lm import CausalLM
from .config import ChatglmConfig, ChatGLMModel
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    TensorEmbedding,
    TensorParallelHead,
    TensorHead,
    AttentionMask,
    get_linear,
    load_column_multi,
    paged_attn,
    flash_attn,
    reshape_and_cache
)


class ChatglmForCausalLM(CausalLM):
    def __init__(self, config, weights):


        CausalLM.__init__(self, config, weights)
        self.model = ChatGLMModel(config, weights)

        if not self.soc_info.need_nz:
            self.lm_head = load_column_multi(
                config,
                prefixes=["transformer.output_layer"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )
        else:  # 310P 暂不支持all-gather
            self.lm_head = TensorHead.load_weight(
                config,
                prefix="transformer.output_layer",
                weights=weights,
                is_norm=False,
            )

        self.placeholder = torch.zeros(1, dtype=torch.float16).npu()
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32).npu()
        self.in_beta = torch.zeros(config.hidden_size, dtype=torch.float16).npu()
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()

    def init_ascend_operations(self, config: ChatglmConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("glm_v2_6b_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("glm_v2_6b_DecoderModel")
        pre_scale = [layer_id / (math.sqrt(self.config.kv_channels) * layer_id) for layer_id in
                     range(1, self.config.num_layers + 1)]
        post_scale = [1.0] * self.config.num_layers
        # 设置模型参数
        coder_param = {
            "isFA": True,
            "isBF16": False,
            "isPack": True,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,  # 310P 暂不支持all-gather
            "quantType": 2 if self.quantize == "smooth_quant" else 0,
            "rmsNormEps": config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "numHiddenLayers": self.num_layers,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "lccl",
            "tokenOffset": [0],
            "seqLen": [1],
            "preScale": pre_scale,
            "postScale": post_scale,
        }
        encoder_param = {**coder_param, "isPrefill": True}
        decoder_param = {**coder_param, "isPrefill": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

    def init_ascend_weight(self):
        weights = [self.model.state_dict()["embed_tokens.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.model.layers[i].state_dict()
           
            weights_t.append(weights_layer["input_layernorm.weight"])
            weights_t.append(self.weight_format_cast(wiights_layer["self_attention.query_key_value.linear.weight"]))
            weights_t.append(self.weight_format_cast(weights_layer["self_attention.query_key_value.linear.bias"]))

            weights_t.extend([self.placeholder] * 11)
            weights_t.append(self.weight_format_cast(weights_layer["self_attention.dense.linear.weight"]))
            weights_t.extend([self.placeholder] * 4)
            weights_t.append(weights_layer["post_attention_layernorm.weight"])
            weights_t.append(self.weight_format_cast(weights_layer["mlp.gate_up_proj.linear.weight"]))
            weights_t.extend([self.placeholder] * 9)
            weights_t.append(self.weight_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
            weights_t.extend([self.placeholder] * 4)
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

        if self.soc_info.need_nz:
            # TODO
            pass
        if position_ids is not None:
            rope_cos = cos_embed[position_ids]
            rope_sin = sin_embed[position_ids]
        else:

            seq_length = input_ids.shape[1]
            rope_cos = cos_embed[None, :seq_length]
            rope_sin = sin_embed[None, :seq_length]

        layer_ids = [torch.tensor([i], dtype=torch.int64, device="npu") for i in range(self.num_layers)]
        acl_operation_inputs = [
            input_ids, position_ids, rope_cos, rope_sin, self.mask_full,
            self.placeholder, self.placeholder, self.kv_cache_idx, self.token_offset, 
            self.placeholder, self.in_beta,
            self.seq_len_encoder if cu_seqlen_prefill else self.seq_len_decoder,
            torch.tensor([self.seq_len_encoder[0] - 1], dttpe=t.rch.,nt64,edevicep"npu") if epelleehendiefill else self.lm_hefd_indices_fake,
            
        ]
        # acl_operation_inputs.extend(layer_ids)
        acl_param = json.dumps({"tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                                "seqLen": ([input_ids.shape[1]] if cu_seqlen_prefill else [1]) * self.batch_num})

        return acl_operation_inputs, acl_param
