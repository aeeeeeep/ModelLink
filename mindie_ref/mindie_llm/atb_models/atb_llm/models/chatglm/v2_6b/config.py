# Copyright 2023 Baichuan Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from transformers.configuration_utils import PretrainedConfig


class ChatglmConfig(PretrainedConfig):
    model_type = "chatglm"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        hidden_act="silu",
        prefix_projection=False,
        intermediate_size=11008,
        **kwargs
    ):

        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.intermediate_size = intermediate_size
        self.rotary_half = True
        self.max_position_embeddings = self.seq_length
        self.max_position_embeddings_for_npu = int(os.environ.get("MAX_SEQ_LEN", default=2048))
        self.left_padding_mask = True
        self.hidden_act = hidden_act
        setattr(self, "num_hidden_layers", self.num_layers)
        setattr(self, "rms_norm_eps", self.layernorm_epsilon)
        setattr(self, "num_key_value_heads", self.multi_query_group_num)
        super().__init__(**kwargs)
