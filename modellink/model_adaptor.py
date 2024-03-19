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

import megatron
from .model import (
    GPTModel, parallel_transformer_init, seq_length_wrapper,
    norm_wrapper, SwitchMLP, state_dict_for_save_checkpoint_wrapper,
    core_attention_wrapper, core_attention_forward, FlashSelfAttention,
    ParallelAttention_wrapper
)
from .core import (vocab_embedding_wrapper, initialize_model_parallel_decorator,
                   destroy_model_parallel_decorator, get_expert_parallel_group,
                   get_expert_parallel_rank, get_expert_model_parallel_rank,
                   get_expert_parallel_world_size, get_expert_model_parallel_world_size,
                   set_expert_model_parallel_rank, set_expert_model_parallel_world_size)
from .data import build_pretraining_data_loader
from .tokenizer import build_tokenizer
from .arguments import parse_args_decorator
from .training import get_model_wrapper
from .utils import ALL_MODULE_WRAPPER_CLASSNAMES
from .checkpointing import _load_base_checkpoint_wrapper, load_checkpoint_wrapper
from .core.datasets.gpt_dataset import _build_document_sample_shuffle_indices
from .initialize import _compile_dependencies


def exe_adaptor():
    import megatron
    megatron.utils.ALL_MODULE_WRAPPER_CLASSNAMES = ALL_MODULE_WRAPPER_CLASSNAMES
    megatron.initialize.parse_args = parse_args_decorator(megatron.initialize.parse_args)
    megatron.initialize._compile_dependencies = _compile_dependencies
    megatron.arguments.parse_args = parse_args_decorator(megatron.arguments.parse_args)
    megatron.global_vars.build_tokenizer = build_tokenizer


    import megatron.training
    megatron.training.get_model = get_model_wrapper(megatron.training.get_model)
    megatron.training.build_pretraining_data_loader = build_pretraining_data_loader

    megatron.model.GPTModel = GPTModel
    megatron.model.transformer.SwitchMLP = SwitchMLP
    megatron.model.transformer.ParallelTransformer.__init__ = parallel_transformer_init
    megatron.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint \
        = state_dict_for_save_checkpoint_wrapper(
        megatron.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint)
    megatron.model.language_model.TransformerLanguageModel.forward = (seq_length_wrapper(
        megatron.model.language_model.TransformerLanguageModel.forward))

    megatron.model.transformer.ParallelAttention.__init__ = ParallelAttention_wrapper(
        megatron.model.transformer.ParallelAttention.__init__)
    megatron.model.transformer.CoreAttention.__init__ = core_attention_wrapper(
        megatron.model.transformer.CoreAttention.__init__)
    megatron.model.transformer.CoreAttention.forward = core_attention_forward
    megatron.model.transformer.FlashSelfAttention = FlashSelfAttention


    from megatron.core.datasets.gpt_dataset import GPTDataset
    GPTDataset._build_document_sample_shuffle_indices = _build_document_sample_shuffle_indices
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = vocab_embedding_wrapper(
        megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward)
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__ = norm_wrapper(
        megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__)

    set_moe_attr()
    megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorator(
        megatron.core.parallel_state.initialize_model_parallel)
    megatron.core.parallel_state.destroy_model_parallel = destroy_model_parallel_decorator(
        megatron.core.parallel_state.destroy_model_parallel)
    megatron.core.mpu = megatron.core.parallel_state

    megatron.checkpointing._load_base_checkpoint = _load_base_checkpoint_wrapper(
        megatron.checkpointing._load_base_checkpoint)
    megatron.training.load_checkpoint = load_checkpoint_wrapper(
        megatron.checkpointing.load_checkpoint)


def set_moe_attr():
    setattr(megatron.core.parallel_state,
            "get_expert_parallel_group", get_expert_parallel_group)
    setattr(megatron.core.parallel_state,
            "get_expert_parallel_rank", get_expert_parallel_rank)
    setattr(megatron.core.parallel_state,
            "get_expert_model_parallel_rank", get_expert_model_parallel_rank)
    setattr(megatron.core.parallel_state,
            "get_expert_parallel_world_size", get_expert_parallel_world_size)
    setattr(megatron.core.parallel_state,
            "get_expert_model_parallel_world_size", get_expert_model_parallel_world_size)
    setattr(megatron.core.parallel_state,
            "set_expert_model_parallel_rank", set_expert_model_parallel_rank)
    setattr(megatron.core.parallel_state,
            "set_expert_model_parallel_world_size", set_expert_model_parallel_world_size)
