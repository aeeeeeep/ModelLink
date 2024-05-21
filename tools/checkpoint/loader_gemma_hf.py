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


def load_args_from_checkpoint_gemma(args):
    # Read Gemma args.
    gemma_args_path = os.path.join(args.load, "config.json")
    with open(gemma_args_path) as f:
        gemma_args = json.load(f)

    # Update Megatron args.
    args.seq_length = 8192
    args.max_position_embeddings = gemma_args["max_position_embeddings"]
    args.hidden_size = gemma_args["hidden_size"]
    args.num_attention_heads = gemma_args["num_attention_heads"]
    args.num_layers = gemma_args["num_hidden_layers"]
    args.global_batch_size = 64
    args.norm_epsilon = gemma_args["rms_norm_eps"]
    args.kv_channels = gemma_args["head_dim"]
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = True
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    args.tokenizer_type = "PretrainedFromHF"
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = False
    args.vocab_size = gemma_args["vocab_size"]
    args.padded_vocab_size = gemma_args["vocab_size"]
    args.llama = gemma_args
    args.ffn_hidden_size = gemma_args["intermediate_size"]
    args.gradient_accumulation_fusion = False
    args.make_vocab_size_divisible_by = 1

    if args.add_dense_bias:
        args.skip_bias_add = False

    if "num_key_value_heads" in gemma_args \
            and gemma_args["num_attention_heads"] != gemma_args["num_key_value_heads"]:
        args.group_query_attention = True
        args.num_query_groups = gemma_args["num_key_value_heads"]


def load_args_from_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(args):
        args_path = os.path.join(args.load, "config.json")
        with open(args_path) as f:
            margs = json.load(f)
        model_type = margs.get('model_type')
        if model_type == 'gemma':
            return load_args_from_checkpoint_gemma(args)
        return fn(args)

    return wrapper


def set_postprocess_state_gemma(args, model, hf_model):
    '''Set embedding params.'''
    model.language_model.encoder.final_norm.weight.data.copy_(hf_model.model.norm.weight)


def set_postprocess_state_wrapper(fn):
    @wraps(fn)
    def wrapper(args, model, hf_model):
        if args.llama['model_type'] == 'gemma':
            return set_postprocess_state_gemma(args, model, hf_model)
        return fn(args)

    return wrapper


def add_arguments(parser):
    loader_llama2.add_arguments(parser)


def load_checkpoint(queue, args):
    loader_llama2.load_args_from_checkpoint \
        = load_args_from_checkpoint_wrapper(loader_llama2.load_args_from_checkpoint)
    loader_llama2.set_postprocess_state \
        = set_postprocess_state_wrapper(loader_llama2.set_postprocess_state)
    try:
        loader_llama2._load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
