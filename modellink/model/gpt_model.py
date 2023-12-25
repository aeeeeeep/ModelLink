# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT-2 model."""

import torch
from modellink.model.module import MegatronModuleForCausalLM
from megatron.model import GPTModel

class BaseModel(GPTModel, MegatronModuleForCausalLM):
    def __init__(self, config, num_tokentypes=0,parallel_output=True,pre_process=True,post_process=True):
        super(BaseModel, self).__init__(config=config,num_tokentypes=num_tokentypes,parallel_output=parallel_output,pre_process=pre_process,post_process=post_process)