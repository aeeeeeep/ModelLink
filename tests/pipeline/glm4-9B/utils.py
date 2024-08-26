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
    
    convert_ckpt_param = config_file["CONVERT_CKPT_PARAM"]
    network_size = config_file["NETWORK_SIZE"]
    tokenizer_param = config_file["TOKENIZER_PARAM"]
    distributed_param = config_file["DISTRIBUTED_PARAM"]
    inference_param = config_file["INFERENCE_PARAM"]
    evaluation_param = config_file["EVALUATION_PARAM"]
    auxiliary_param = config_file["AUXILIARY_PARAM"]

    inference_hf_chat_param = config_file["INFERENCE_HF_CHAT_PARAM"]
    inference_prompt_chat_param = config_file["INFERENCE_PROMPT_CHAT_PARAM"]


def assert_judge(expression):
    if not expression:
        raise AssertionError
