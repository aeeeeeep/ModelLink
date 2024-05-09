import json
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParamConfig:
    base_dir = Path(__file__).absolute().parent
    param_config = os.path.join(base_dir, "param_config.json")
    with open(param_config) as f:
        config_file = json.load(f)

    distributed_param = config_file["DISTRIBUTED_PARAM"]
    network_size = config_file["NETWORK_SIZE"]
    inference_param = config_file["INFERENCE_PARAM"]
    evaluation_param = config_file["EVALUATION_PARAM"]
    auxiliary_param = config_file["AUXILIARY_PARAM"]
    tokenizer_param = config_file["TOKENIZER_PARAM"]
    lora_inference_param = config_file["LORA_INFERENCE_PARAM"]
    convert_weight_param = config_file["CONVERT_WEIGHT_PARAM"]


def assert_judge(expression):
    if not expression:
        raise AssertionError
