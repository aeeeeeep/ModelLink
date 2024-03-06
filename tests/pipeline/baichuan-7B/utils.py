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
    
    pretrain_data_param = config_file["PROCESS_PRETRAIN_DATA"]
    instruction_data_param = config_file["PROCESS_INSTRUCTION_DATA"]


def assert_judge(expression):
    if not expression:
        raise AssertionError
