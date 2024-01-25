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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
from functools import wraps
import loader_llama2_hf as loader_llama2


def load_args_from_checkpoint_bloom(args):
    # Read Llama args.
    llama_args_path = os.path.join(args.load, "config.json")
    with open(llama_args_path) as f:
        llama_args = json.load(f)

    # Update Megatron args.
    args.seq_length = 4096
    args.max_position_embeddings = 4096
    args.hidden_size = llama_args["hidden_size"]
    args.num_attention_heads = llama_args["n_head"]
    args.num_layers = llama_args["n_layer"]
    args.global_batch_size = 1024
    args.norm_epsilon = llama_args["layer_norm_epsilon"]
    args.iteration = 1 # '0', 'release' don't work
    args.add_position_embedding = True
    args.use_rotary_position_embeddings = True
    args.swiglu = False
    args.tokenizer_type = "Llama2Tokenizer"
    args.fp16 = True
    args.normalization = "LayerNorm"
    args.add_bias_linear = True
    args.untie_embeddings_and_output_weights = False
    args.vocab_size = llama_args["vocab_size"]
    args.padded_vocab_size = llama_args["vocab_size"]
    args.llama = llama_args
    args.ffn_hidden_size = 16384
    args.gradient_accumulation_fusion = False

    if "num_key_value_heads" in llama_args:
        args.group_query_attention = True
        args.num_query_groups = llama_args["num_key_value_heads"]


def load_checkpoint_to_model_bloom(args):
    '''Set model params.'''
    from pretrain_gpt import model_provider
    from transformers import BloomForCausalLM

    # Load Huggingface model.
    hf_model = BloomForCausalLM.from_pretrained(args.load, device_map="cpu")

    # Init Megatron model.
    model = model_provider(True, True).to(args.params_dtype)

    for name_param_h, name_param_m in zip(hf_model.named_parameters(), model.named_parameters()):
        name_h, param_h = name_param_h
        name_m, param_m = name_param_m
        param_m.data.copy_(param_h)

    return model


def load_args_from_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(args):
        args_path = os.path.join(args.load, "config.json")
        with open(args_path) as f:
            margs = json.load(f)
        model_type = margs.get('model_type')
        if model_type == 'bloom':
            return load_args_from_checkpoint_bloom(args)
        return fn(args)

    return wrapper


def load_checkpoint_to_model_wrapper(fn):
    @wraps(fn)
    def wrapper(args):
        if args.llama['model_type'] == 'bloom':
            return load_checkpoint_to_model_bloom(args)
        return fn(args)

    return wrapper


def add_arguments(parser):
    loader_llama2.add_arguments(parser)


def load_checkpoint(queue, args):
    loader_llama2.load_args_from_checkpoint \
        = load_args_from_checkpoint_wrapper(loader_llama2.load_args_from_checkpoint)
    loader_llama2.load_checkpoint_to_model \
        = load_checkpoint_to_model_wrapper(loader_llama2.load_checkpoint_to_model)
    try:
        loader_llama2._load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise

