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
import json
import logging
import torch
import torch_npu

from common.utils import init_resource
import examples

_TO_TORCH_TENSOR = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16
}


def update_model_config(config):
    config["local_rank"] = int(os.getenv("LOCAL_RANK", "0"))
    config["world_size"] = int(os.getenv("WORLD_SIZE", "1"))
    config["device_list"] = list(map(lambda x: int(x), config["device_list"].split(",")))


if __name__ == "__main__":
    with open("config/llm_inference.json", "r") as cf:
        model_config = json.load(cf)

    update_model_config(model_config)

    if model_config["model_path"] == "":
        logging.error("model_path can not be empty")
        exit(1)

    init_resource(model_config)

    device_list = model_config["device_list"]
    local_rank = model_config["local_rank"]
    is_logging = (local_rank == 0 or local_rank == device_list[0])
    if is_logging:
        logging.info("Model execution configuration is: \n%s", json.dumps(model_config, indent=4))

    model_config["dtype"] = _TO_TORCH_TENSOR[model_config["dtype"]]

    run_model = getattr(examples, "run_%s" % model_config["model"])
    result = run_model(model_config)

    if (result == 0) and is_logging:
        logging.info("model run success")