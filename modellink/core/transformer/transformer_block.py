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
from megatron.core import parallel_state
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
