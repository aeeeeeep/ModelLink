# Copyright 2022 EleutherAI and the HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import torch

try:
    import torch_npu
except ImportError:
    pass
import torch.nn as nn

from transformers.activations import ACT2FN
from configuration_internlm import InternLMConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

class GraphOpt:
    @staticmethod
    def set_module(model,
                   submodule_key,
                   module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)

class InternLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        InternLMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states  + self.bias

class InternLMMLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def replace_module(model, config: InternLMConfig):
    """
    离群点抑制后，model结构适配

    :param model:
    :param config:
    :return:
    """
    for name, module in model.named_modules():
        world_size = 1
        if hasattr(config, 'world_size'):
            world_size = config.world_size

        if "norm" in name:
            new_module = InternLMRMSNorm(module.weight.size(0), module.variance_epsilon)
            new_module.weight.data = new_module.weight.data.type(module.weight.data.dtype)
            new_module.bias.data = new_module.bias.data.type(module.weight.data.dtype)
            new_module.to(module.weight.data.device)
            GraphOpt.set_module(model, name, new_module)

        elif "mlp.gate_proj" in name:
            new_module = nn.Linear(config.hidden_size, config.intermediate_size // world_size, bias=True)
            new_module.weight.data = new_module.weight.data.type(module.weight.data.dtype)
            new_module.bias.data = new_module.bias.data.type(module.weight.data.dtype)
            new_module.to(module.weight.data.device)
            GraphOpt.set_module(model, name, new_module)

        elif "mlp.up_proj" in name:
            new_module = nn.Linear(config.hidden_size, config.intermediate_size // world_size, bias=True)
            new_module.weight.data = new_module.weight.data.type(module.weight.data.dtype)
            new_module.bias.data = new_module.bias.data.type(module.weight.data.dtype)
            new_module.to(module.weight.data.device)
            GraphOpt.set_module(model, name, new_module)

        elif "mlp.down_proj" in name:
            new_module = nn.Linear(config.intermediate_size // world_size, config.hidden_size, bias=False)
            new_module.weight.data = new_module.weight.data.type(module.weight.data.dtype)
            new_module.to(module.weight.data.device)
            GraphOpt.set_module(model, name, new_module)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="replace_module.")
    parser.add_argument(
        "--input_path",
        default="./",
        help="Location of Model weights, which contains model folders",
    )
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.input_path, trust_remote_code=True, torch_dtype=torch.float16)
    replace_model = replace_module(model, model.config)
    print(replace_model)

