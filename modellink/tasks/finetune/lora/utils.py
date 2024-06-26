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

import re
from typing import List

from megatron.training import get_args


def get_lora_model_classes():
    from peft import LoraModel, PeftModel
    return PeftModel, LoraModel


def is_enable_lora():
    args = get_args()
    if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
        return True
    return False


def merge_dicts(dict1, dict2):
    result = dict1
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def modify_keys_with_dict(dictionary, words_to_replace, exclude_words):
    args = get_args()
    layers_to_transform = list(map(str, args.lora_layers_to_transform))
    target_modules = args.lora_target_modules

    modified_dict = {}
    for key, value in dictionary.items():
        key_str = str(key)
        
        # Check if a word to replace is present in the key
        word_to_replace = next((word for word in words_to_replace if word in key_str), None)
        should_be_replace = word_to_replace and word_to_replace != key_str

        # Check if a word none of the exclude_words are present
        is_excluded = any(exclude_word in key_str for exclude_word in exclude_words)

        if (
            should_be_replace
            and not is_excluded
            # If the layers is not affected by LoRA - it mustn't be modified in the state dict
            and _lora_affected_layer(key_str, target_modules, layers_to_transform)
        ):
            key_str = key_str.replace(word_to_replace, words_to_replace[word_to_replace])

        if isinstance(value, dict):
            modified_dict[key_str] = modify_keys_with_dict(value, words_to_replace, exclude_words)
        else:
            modified_dict[key_str] = value
    return modified_dict


def _lora_affected_layer(key, target_modules: List[str], layers_to_transform: List[int]) -> bool:
    if not layers_to_transform:
        return True
    layers_with_lora_pattern = '|'.join(target_modules)
    layer_indicies_for_lora_pattern = '|'.join(layers_to_transform)
    lora_adapter_name_pattern = re.compile(
        rf'^.*\.({layer_indicies_for_lora_pattern})\..*\.({layers_with_lora_pattern}).*$'
    )
    match = lora_adapter_name_pattern.match(key)
    return bool(match)