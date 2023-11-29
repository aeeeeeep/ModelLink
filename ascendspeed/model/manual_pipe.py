# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION. All rights reserved.
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

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
import deepspeed.runtime.utils as ds_utils
from deepspeed.utils import logger
import torch.nn as nn
from ascendspeed import get_args
from ascendspeed.enums import TransformerLayerStage
from ascendspeed.model.gpt_model import ParallelTransformerLayerPipe


def bloom_manual_layer_allocation():
    args = get_args()
    parts = [0]
    for nlayer in args.manual_layers:
        parts.append(parts[-1] + nlayer)
    for i in range(1, len(parts)):
        parts[i] += 3

    parts[-1] += 4
    return parts


def use_bloom_manual_layer_allocation():
    args = get_args()
    return args.use_manual_layer_allocation


class ManuallyAllocatedPipelineModule(PipelineModule):
    def _partition_layers(self, method='uniform'):
        super(ManuallyAllocatedPipelineModule, self)._partition_layers(method)
        stage_id = self._topo.get_coord(self.global_rank).pipe
        method = method.lower()
        if method.startswith('type:'):
            if use_bloom_manual_layer_allocation():
                parts = bloom_manual_layer_allocation()
                self._set_bounds(start=parts[stage_id], stop=parts[stage_id + 1])
    
    def forward(self, forward_input):
        if get_args().communication_slim < 2:
            return super().forward(forward_input)
        if self.activation_checkpoint_interval > 1:
            raise ValueError("`--communication-slim` does not support " 
            "for `activation_checkpoint_interval` > 1")
        self.micro_offset += 1
        
        def exec_range_func(start, end):
            local_micro_offset = self.micro_offset + 1
            
            def exec_func(*inputs):
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (self.base_seed * local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)
                    inputs = layer(inputs)
                return inputs
            return exec_func

        def exec_nocomm_func(start, end):
            local_micro_offset = self.micro_offset + 1
            
            def exec_attn(*inputs):
                if len(inputs) == 1:
                    inputs = inputs[0]
                layer = self.forward_funcs[start]
                if self.seed_layers:
                    new_seed = (self.base_seed * local_micro_offset) + self.curr_layer
                    if self.seed_fn:
                        self.seed_fn(new_seed)
                    else:
                        ds_utils.set_random_seed(new_seed)
                inputs = layer(inputs, transformer_stage=TransformerLayerStage.attn)
                return inputs

            def exec_ffn(*inputs):
                if len(inputs) == 1:
                    inputs = inputs[0]
                layer = self.forward_funcs[start]
                inputs = layer(inputs, transformer_stage=TransformerLayerStage.ffn)
                return inputs
            return exec_attn, exec_ffn
       
        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            x = func(forward_input)
        else:
            num_layers = len(self.forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(start_idx + self.activation_checkpoint_interval, num_layers)

                funcs = self.forward_funcs[start_idx:end_idx]
                if not isinstance(x, tuple):
                    x = (x, )
                if self._is_checkpointable(funcs):
                    if isinstance(funcs[0], ParallelTransformerLayerPipe):
                        attn_func, ffn_func = exec_nocomm_func(start_idx, end_idx)
                        output = self.activation_checkpoint_func(attn_func, *x)
                        if not isinstance(output, tuple):
                            output = (output, )
                        x = self.activation_checkpoint_func(ffn_func, *output)
                    else:
                        x = self.activation_checkpoint_func(exec_range_func(start_idx, end_idx), *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x