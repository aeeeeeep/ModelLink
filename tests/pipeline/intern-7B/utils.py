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
        lora_param,
        training_param,
        training_auxiliary,
        learning_rate,
        regularization,
        and other auxiliary_param.
    """
    base_dir = Path(__file__).absolute().parent
    param_config = os.path.join(base_dir, "param_config.json")
    with open(param_config) as f:
        config_file = json.load(f)
    
    distributed_param = config_file["DISTRIBUTED_PARAM"]
    network_size = config_file["NETWORK_SIZE"]
    inference_param = config_file["INFERENCE_PARAM"]
    evaluation_param = config_file["EVALUATION_PARAM"]
    training_param = config_file["TRAINING_PARAM"]
    training_aux = config_file["TRAINING_AUX"]
    learning_rate_param = config_file["LEARNING_RATE"]
    regularization = config_file["REGULARIZATION"]
    auxiliary_param = config_file["AUXILIARY_PARAM"]
    process_pretrain_data = config_file["PROCESS_PRETRAIN_DATA"]
    inference_aux = config_file["INFERENCE_AUX"]
    convert_ckpt_param = config_file["CONVERT_CKPT_PARAM"]

    
def assert_judge(expression):
    if not expression:
        raise AssertionError