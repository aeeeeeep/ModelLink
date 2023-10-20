# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from ascendspeed import get_args
from ascendspeed.model import LlamaModel
from ascendspeed.initialize import initialize_megatron
from ascendspeed.arguments import core_transformer_config_from_args
from tasks.inference.infer_base import task1, task2, task3, task4, task5, task6, add_text_generate_args


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    config = core_transformer_config_from_args(get_args())
    init_model = LlamaModel(
        config,
        parallel_output=False,
        add_pooler=False,
        pre_process=pre_process,
        post_process=post_process
    )
    return init_model


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    model = LlamaModel.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    system_template = "Below is an instruction that describes a task, paired with an input that provides further " \
                      "context. Write a response that appropriately completes the request. " \
                      "Please note that you need to think through your response logically and step by step.\n\n"
    dialog_template = "### Instruction:\n{instruction}\n\n### Response:"
    template = system_template + dialog_template

    task1(args, model, system_template=system_template, dialog_template=dialog_template)
    task2(args, model, system_template=system_template, dialog_template=dialog_template)
    task3(args, model, system_template=system_template, dialog_template=dialog_template)
    task4(args, model, system_template=system_template, dialog_template=dialog_template)
    task5(args, model, system_template=system_template, dialog_template=dialog_template)
    task6(args, model, system_template=system_template, dialog_template=dialog_template)
