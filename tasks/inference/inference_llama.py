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

from ascendspeed import megatron_adaptor

import modellink
from megatron import get_args
from megatron.model import GPTModel
from megatron.initialize import initialize_megatron
from megatron.arguments import core_transformer_config_from_args
from tasks.inference.text_generation.infer_base import task_factory, add_text_generate_args


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    config = core_transformer_config_from_args(get_args())

    init_model = GPTModel(
        config,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )
    return init_model


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    model = GPTModel.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    task_factory(args, model)
