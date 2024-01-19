# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
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

"""Pretrain InternLM"""
import os
from functools import partial
from typing import Union

import torch
import torch_npu
from torch import Tensor
from ascendspeed import megatron_adaptor
#from modellink.adaptor_model import apply_Internlm_patch
from torch_npu.contrib import transfer_to_npu
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset
import megatron.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.utils import (
    get_ltor_masks_and_position_ids,
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    gpt_layer_with_transformer_engine_spec_moe
)
from pretrain_gpt import (
    get_batch,
    loss_func,
    forward_step,
    is_dataset_built_on_rank,
    core_gpt_dataset_config_from_args,
    train_valid_test_datasets_provider
)
import modellink
def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0('building InternLM model ...')
    config = core_transformer_config_from_args(get_args())
    config.column_parallel_linear_bias = True
    config.row_parallel_linear_bias = True
    config.row_parallel_linear_skip_bias_add = False
    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts is None:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
            else:
                transformer_layer_spec = gpt_layer_with_transformer_engine_spec_moe

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return model


if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=False)
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
