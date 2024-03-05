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

import logging
import torch

from common.utils import *
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import LlamaTokenizer


def generate_tokenizer(model_config):
    model_path = model_config["model_path"]
    input_padding = model_config["input_padding"]

    if input_padding:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)

    return tokenizer


def generate_model(model_config):
    model_path = model_config["model_path"]
    dtype = model_config["dtype"]
    backend = model_config["backend"]
    distributed_mode = model_config["distributed_mode"]
    exe_mode = model_config["exe_mode"]
    local_rank = model_config["local_rank"]
    world_size = model_config["world_size"]
    device_list = model_config["device_list"]

    logging.info("Set execution using %s index: %s", backend, local_rank)
    device = torch.device("%s:%s" % (backend, local_rank))

    if backend == "npu":
        torch.npu.set_device(device)

    is_logging = (len(device_list) > 1 and (local_rank == 0 or local_rank == device_list[0])) or (len(device_list) == 1)
    if is_logging:
        logging.info("Try to load pretrained model in path: %s", model_path)

    model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=dtype)
    model.world_size = world_size

    tokenizer = generate_tokenizer(model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if distributed_mode == "deepspeed":
        import deepspeed
        if backend == "npu":
            deepspeed.init_distributed(dist_backend="hccl")
            if exe_mode == "dynamo":
                import torch_npu
                import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce

        model = deepspeed.init_inference(
            model=model,
            mp_size=world_size,
            dtype=dtype,
            replace_with_kernel_inject=False,
            injection_policy={LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
        )
    model.to(device)
    logging.info("The final model structure is: \n %s", model)

    if exe_mode == "dynamo":
        torch._dynamo.reset()

    model_config["model"] = model
    model_config["tokenizer"] = tokenizer
    model_config["device"] = device


def run_llama2(model_config):

    set_options(model_config)

    generate_prompts(model_config)

    generate_model(model_config)

    generate_answer(model_config)

    return 0