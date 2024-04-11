# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

from .tensor_parallel.layers import vocab_embedding_wrapper
from .parallel_state import (initialize_model_parallel_decorator, destroy_model_parallel_decorator,
                             get_expert_model_parallel_rank,
                             get_expert_model_parallel_world_size, get_expert_parallel_group,
                             get_expert_parallel_rank, get_expert_parallel_world_size,
                             set_expert_model_parallel_world_size, set_expert_model_parallel_rank)
from .models.common.embeddings.rotary_pos_embedding import RotaryEmbedding_forward_wrapper, apply_rotary_pos_emb_wrapper
