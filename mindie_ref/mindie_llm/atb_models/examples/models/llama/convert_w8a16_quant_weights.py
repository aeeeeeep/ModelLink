# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
import argparse
import shutil
import stat

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16_model_path',
                        help="model and tokenizer path",
                        default='/data/acltransformer_testdata/weights/llama2/llama-2-70b',
                        )
    parser.add_argument('--w8a16_model_path',
                        help="model and tokenizer path",
                        default='/data/acltransformer_testdata/weights/llama2/llama-2-70b_w8a16',
                        )
    return parser.parse_args()


def convert_2_w8a16_weight(fp16_model_path, w8a16_model_path):
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=fp16_model_path, trust_remote_code=True).float().cpu()

    # w_sym=True：对称量化，w_sym=False：非对称量化;
    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        dev_type="cpu",
        act_method=3,
        pr=1.0,
        w_sym=True,
        mm_tensor=False
    )

    calibrator = Calibrator(model, quant_config, calib_data=None, disable_level='L0')
    calibrator.run()
    calibrator.save(w8a16_model_path, save_type=["safe_tensor"])


def copy_config_tokenizer(fp16_model_path, w8a16_model_path):
    with open(os.path.join(fp16_model_path, "config.json"), 'r') as f:
        config = json.load(f)
    config['quantize'] = 'w8a16'

    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(w8a16_model_path, "config.json"), flags, modes), "w") as f:
        json.dump(config, f)

    shutil.copyfile(os.path.join(fp16_model_path, "tokenizer_config.json"), os.path.join(w8a16_model_path, "tokenizer_config.json"))
    shutil.copyfile(os.path.join(fp16_model_path, "tokenizer.json"), os.path.join(w8a16_model_path, "tokenizer.json"))
    shutil.copyfile(os.path.join(fp16_model_path, "tokenizer.model"), os.path.join(w8a16_model_path, "tokenizer.model"))


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(args.w8a16_model_path):
        os.mkdir(args.w8a16_model_path)

    convert_2_w8a16_weight(args.fp16_model_path, args.w8a16_model_path)

    copy_config_tokenizer(args.fp16_model_path, args.w8a16_model_path)
