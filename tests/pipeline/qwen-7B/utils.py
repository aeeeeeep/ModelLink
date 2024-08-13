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
    distributed_param_tp8_pp1 = config_file["DISTRIBUTED_PARAM_TP8_PP1"]
    auxiliary_param = config_file["AUXILIARY_PARAM"]
    instruction_param = config_file["INSTRUCTION_PARAM"]
    output_param = config_file["OUTPUT_PARAM"]

    # prepreocess instruction data
    instruction_data_param = config_file["PROCESS_INSTRUCTION_DATA"]
    instruction_data_mix_param1 = config_file["PROCESS_INSTRUCTION_DATA_MIX1"]
    instruction_data_mix_param2 = config_file["PROCESS_INSTRUCTION_DATA_MIX2"]

    # inference
    inference_param = config_file["INFERENCE_PARAM"]
    beam_search_auxliary_param = config_file["BEAM_SEARCH_AUXILIARY_PARAM"]
    greedy_search_auxliary_param = config_file["GREEDY_SEARCH_AUXILIARY_PARAM"]
    do_sample_auxliary_param = config_file["DO_SAMPLE_AUXILIARY_PARAM"]
    beam_search_with_sampling_auxliary_param = config_file["BEAM_SEARCH_WITH_SAMPLING_AUXILIARY_PARAM"]
    return_output_log_probs_auxliary_param = config_file["RETURN_OUTPUT_LOG_PROBS_AUXILIARY_PARAM"]


def assert_judge(expression):
    if not expression:
        raise AssertionError
