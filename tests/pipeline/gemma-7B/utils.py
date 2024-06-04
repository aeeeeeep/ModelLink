import json
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParamConfig:
    """
    We can config the params in the `.json` file including: 
        distributed_param,
        network_size,
        inference_param,
        evaluation_param,
        and other auxiliary_param.
    """
    base_dir = Path(__file__).absolute().parent
    param_config = os.path.join(base_dir, "param_config.json")
    with open(param_config) as f:
        config_file = json.load(f)
    
    distributed_param = config_file["DISTRIBUTED_PARAM"]
    network_size = config_file["NETWORK_SIZE"]
    inference_aux = config_file["INFERENCE_AUX"]
    inference_param = config_file["INFERENCE_PARAM"]
    evaluation_param = config_file["EVALUATION_PARAM"]
    auxiliary_param = config_file["AUXILIARY_PARAM"]
    pretrain_data_param = config_file["PROCESS_PRETRAIN_DATA"]
    convert_ckpt_param = config_file["CONVERT_CKPT_FROM_HF"]


def assert_judge(expression):
    if not expression:
        raise AssertionError
