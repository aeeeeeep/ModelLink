#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

import json
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParamConfig:
    """
    We can config the params in the `.json` file including:
        convert_ckpt_param,
        network_size,
        tokenizer_param,
        distributed_param,
        inference_param,
        evaluation_param,
        and other auxiliary_param.
    """
    base_dir = Path(__file__).absolute().parent
    param_config = os.path.join(base_dir, "param_config.json")
    with open(param_config) as f:
        config_file = json.load(f)

    network_size = config_file["NETWORK_SIZE"]
    tokenizer_param = config_file["TOKENIZER_PARAM"]
    distributed_param = config_file["DISTRIBUTED_PARAM"]
    auxiliary_param = config_file["AUXILIARY_PARAM"]
    instruction_param = config_file["INSTRUCTION_PARAM"]
    output_param = config_file["OUTPUT_PARAM"]
    instruction_data_param = config_file["PROCESS_INSTRUCTION_DATA"]
    instruction_data_mix_param1 = config_file["PROCESS_INSTRUCTION_DATA_MIX1"]
    instruction_data_mix_param2 = config_file["PROCESS_INSTRUCTION_DATA_MIX2"]


def assert_judge(expression):
    if not expression:
        raise AssertionError
