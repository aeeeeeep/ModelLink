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

from script.llama2 import run_llama2
from common.utils import *

import os
import json
import logging
import torch
import torch_npu

if __name__ == "__main__":
    with open("config/llm_inference.json", "r") as cf:
        model_config = json.load(cf)

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    model_config["local_rank"] = local_rank
    model_config["world_size"] = world_size

    #modify configurations
    device_list = list(map(lambda  x: int(x), model_config["device_list"].split(",")))
    model_config["device_list"] = device_list
    if model_config["model_path"] == "":
        logging.error("model_path can not be empty")
        exit(1)

    init_resource(model_config)

    is_logging = (len(device_list) > 1 and (local_rank == 0 or local_rank == device_list[0])) or (len(device_list) == 1)
    if is_logging:
        logging.info("Model execution configuration is: \n%s", json.dumps(model_config, indent=4))

    if model_config["dtype"] == "fp16":
        model_config["dtype"] = torch.float16
    elif model_config["dtype"] == "fp32":
        model_config["dtype"] = torch.float32
    elif model_config["dtype"] == "bf16":
        model_config["dtype"] = torch.bfloat16
    else:
        model_config["dtype"] = torch.float16

    try:
        result = run_llama2(model_config)
    except Exception as e:
        if is_logging:
            logging.error("model run failed, %s", e)
        result = 1

    if result == 0:
        if is_logging:
            logging.info("model run success")
        exit(0)
    else:
        if is_logging:
            logging.error("model run failed")
        exit(1)