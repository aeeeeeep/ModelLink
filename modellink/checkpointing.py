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

import os

from functools import wraps
from megatron.checkpointing import _load_base_checkpoint
from megatron import get_args
from .tasks.finetune.lora.utils import is_enable_lora, merge_dicts, modify_keys_with_dict


def _load_base_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        state_dict, checkpoint_name, release = fn(*args, **kwargs)
        rank0 = kwargs.pop('rank0')
        if is_enable_lora() and state_dict is not None:
            args_ = get_args()
            words_to_match = {'weight': 'base_layer.weight', 'bias': 'base_layer.bias'}
            exclude_words = ['base_layer', 'lora_', 'norm']
            state_dict = modify_keys_with_dict(state_dict, words_to_match, exclude_words)

            if not args_.lora_load:
                return state_dict, checkpoint_name, release

            # Read the tracker file and set the iteration.
            state_dict_lora, checkpoint_name_lora, release_lora = _load_base_checkpoint(args_.lora_load, rank0)
            if state_dict_lora is not None:
                merge_dicts(state_dict, state_dict_lora)
                checkpoint_name = checkpoint_name_lora
                release = release_lora
        return state_dict, checkpoint_name, release
    return wrapper


def load_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if is_enable_lora() and (not args_.lora_load or not os.path.exists(args_.lora_load)):
            strict = False
            kwargs['strict'] = strict

        return fn(*args, **kwargs)

    return wrapper
