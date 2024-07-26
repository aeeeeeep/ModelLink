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

import torch

from functools import wraps
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.transformer import build_module
from megatron.core.transformer.custom_layers.transformer_engine import TENorm


def get_num_layers_to_build_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        num_layers_to_build = fn(self, *args, **kwargs)
        args = get_args()
        # add args_pos_norm, different with megatron
        if args.num_layer_list:
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                raise ValueError("Dynamic pipeline model and virtual pipeline cannot be enabled at the same time.")
            pp_stage = parallel_state.get_pipeline_model_parallel_rank()
            num_layer_list = list(map(int, args.num_layer_list.split(',')))
            num_layers_to_build = num_layer_list[pp_stage]

        return num_layers_to_build
    return wrapper


def _transformer_block_build_layers(self):
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    def build_layer(layer_spec, layer_number):
        # TODO：补充first_k_dense_replace和moe_layer_freq的参数校验
        if (
                args.num_experts
                and args.first_k_dense_replace is not None
                and args.moe_layer_freq is not None
        ):
            offset = parallel_state.get_pipeline_model_parallel_rank() * len(self.submodules.layer_specs)
            layer_idx = layer_number + offset - 1
            if (
                    layer_idx >= args.first_k_dense_replace
                    and layer_idx % args.moe_layer_freq == 0
            ):
                layer_spec.submodules.mlp = _get_mlp_module_spec(use_te=use_te, num_experts=args.num_experts,
                                                                 moe_grouped_gemm=args.moe_grouped_gemm)
            else:
                layer_spec.submodules.mlp = _get_mlp_module_spec(use_te=use_te, moe_grouped_gemm=args.moe_grouped_gemm)

        return build_module(layer_spec, config=self.config, layer_number=layer_number, )

    # offset is implicit in TransformerLayer
    self.layers = torch.nn.ModuleList(
        [
            build_layer(layer_spec, i + 1)
            for i, layer_spec in enumerate(self.submodules.layer_specs)
        ]
    )

    if self.post_process and self.post_layer_norm:
        # Final layer norm before output.
        self.final_layernorm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
