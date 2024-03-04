# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from ascendspeed import megatron_adaptor

import modellink
from megatron import get_args
from megatron.model import GPTModel
from megatron.initialize import initialize_megatron
from megatron.arguments import core_transformer_config_from_args
from tasks.inference.infer_base import task_factory, add_text_generate_args


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

    system_template = "Below is an instruction that describes a task, paired with an input that provides further " \
                      "context. Write a response that appropriately completes the request. " \
                      "Please note that you need to think through your response logically and step by step.\n\n"
    dialog_template = "### Instruction:\n{instruction}\n\n### Response:"
    template = system_template + dialog_template

    task_factory(args, model, system_template=system_template, dialog_template=dialog_template)
