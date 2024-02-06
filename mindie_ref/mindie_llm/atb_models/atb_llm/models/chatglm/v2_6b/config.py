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
from transformers.configuration_utils import PretrainedConfig


class ChatglmConfig(PretrainedConfig):
    model_type = "chatglm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=65024,
            hidden_size=4096,
            kv_channels=128,
            intermediate_size=11008,
            num_layers=28,
            multi_query_group_num=2,
            num_attention_heads=32,
            hidden_act="silu",
            seq_length=8192,
            initializer_range=0.02,
            layernorm_epsilon=1e-5,
            use_cache=True,
            eos_token_id=2,
            tie_word_embeddings=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.kv_channels = 128
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.multi_query_group_num = multi_query_group_num
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layernorm_epsilon = layernorm_epsilon
        self.use_cache = use_cache

        super().__init__(
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )