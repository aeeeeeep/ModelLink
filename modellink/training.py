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


from functools import wraps

from peft import LoraConfig, get_peft_model
from megatron.arguments import core_transformer_config_from_args
from megatron import get_args
from .tasks.finetune.lora.utils import is_enable_lora


def get_model_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        args = get_args()

        if is_enable_lora():
            config = core_transformer_config_from_args(args)
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=0.0,
                bias="none",
                megatron_config=config,
                megatron_core="megatron.core",
            )

            for model_item in model:
                model_item = get_peft_model(model_item, lora_config)
                model_item.print_trainable_parameters()

        return model
    return wrapper
