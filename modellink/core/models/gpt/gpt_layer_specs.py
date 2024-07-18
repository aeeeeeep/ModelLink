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

from functools import wraps

from megatron.training import get_args
from modellink.core.transformer.custom_layers.transformer_engine import PTNorm


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False):
        res = fn(num_experts, moe_grouped_gemm, qk_layernorm)
        args_pos_norm = get_args()
        res.submodules.input_layernorm = PTNorm
        if qk_layernorm:
            res.submodules.self_attention.submodules.q_layernorm = PTNorm
            res.submodules.self_attention.submodules.k_layernorm = PTNorm
        res.submodules.pre_mlp_layernorm = PTNorm
        if args_pos_norm.post_norm:
            res.submodules.post_attn_norm = PTNorm
            res.submodules.post_mlp_layernorm = PTNorm
        return res

    return wrapper
