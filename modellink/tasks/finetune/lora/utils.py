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

from peft import PeftModel, LoraModel
from megatron import get_args


def get_lora_model_classes():
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
    modified_dict = {}
    for key, value in dictionary.items():
        key_str = str(key)
        matched_word = next((word for word, replacement in words_to_replace.items() if word in key_str), None)
        if (matched_word and
                not any(exclude_word in key_str for exclude_word in exclude_words) and
                key_str != matched_word):
            # Check if a word to replace is present in the key and none of the exclude_words are present
            new_key = key_str.replace(matched_word, words_to_replace[matched_word])
            if isinstance(value, dict):
                modified_dict[new_key] = modify_keys_with_dict(value, words_to_replace, exclude_words)
            else:
                modified_dict[new_key] = value
        else:
            if isinstance(value, dict):
                modified_dict[key] = modify_keys_with_dict(value, words_to_replace, exclude_words)
            else:
                modified_dict[key] = value
    return modified_dict