# coding=utf-8
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
from torch import Tensor
from megatron.core import tensor_parallel, parallel_state, mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args


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


def transformer_block_checkpointed_forward_wrapper(forward_func):
    @wraps(forward_func)
    def block_method_checkpointed_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.recompute_method == 'block':
            output = _block_method_checkpointed_forward_func(*args, **kwargs)
        else:
            output = forward_func(*args, **kwargs)
        return output

    return block_method_checkpointed_forward


def _block_method_checkpointed_forward_func(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
):
    """
        Forward method with activation checkpointing.
        Should only used when recompute_method is 'block'.
        This forward_func is only used for enable_recompute_layers_per_pp_rank.
    """
    def custom(start: int, end: int):
        """
        A provider for original(vanilla) forward function.
        """
        def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
        ):
            for index in range(start, end):
                layer = self._get_layer(index)
                hidden_states, context = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=None,
                    packed_seq_params=packed_seq_params,
                )
            return hidden_states, context

        return custom_forward

    global_args = get_args()
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = global_args.virtual_pipeline_model_parallel_size
    if vpp_rank is None or not global_args.enable_recompute_layers_per_pp_rank:
        vpp_rank = 0
    if vpp_size is None or not global_args.enable_recompute_layers_per_pp_rank:
        vpp_size = 1

    for l in range(self.num_layers_per_pipeline_rank):
        should_recompute = (l * vpp_size + vpp_rank) < self.config.recompute_num_layers
        if should_recompute:
            hidden_states, context = tensor_parallel.checkpoint(
                custom(l, l + 1),
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            )
        else:
            hidden_states, context = custom(l, l + 1)(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            )

    return hidden_states
