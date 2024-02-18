# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import torch
import torch_npu
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BloomTokenizerFast)
MODEL_FILE = os.path.join(model_precision_test.ATB_SPEED_HOME_PATH, "pytorch/examples/bloom7b")
sys.path.append(MODEL_FILE)
from modelling_bloom_ascend import BloomForCausalLM
from base import model_precision_test

WEIGHT_DIR = os.path.join(model_precision_test.ATB_TESTDATA_PATH, "weights", "bloom_7b")


class Bloom7BModelTest(model_precision_test.ModelPrecisionTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_model(self, hardware_type):
        if hardware_type == "NPU":
            class Args:
                pass
            args = Args()
            args.device = [self.device_id, self.device_id + 1]
            args.model_path = WEIGHT_DIR
            args.data_dtype = "fp16"
            args.hardware = "310"

            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            torch_npu.npu.set_device(args.device[local_rank])

            # seed must be the same in all processes
            torch.manual_seed(1)

            tokenizer_path = os.path.join(args.model_path, 'tokenizer')
            tokenizer = BloomTokenizerFast.from_pretrained(tokenizer_path, use_fast=False)
            part_model_path = os.path.join(args.model_path, 'part_model', str(local_rank))
            config = AutoConfig.from_pretrained(part_model_path)
            config.model_path = args.model_path
            config.data_dtype = args.data_dtype
            config.hardware = args.hardware
            model = BloomForCausalLM(config).half().npu()

        else:
            tokenizer = BloomTokenizerFast.from_pretrained(WEIGHT_DIR, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(WEIGHT_DIR, torch_dtype=torch.float16).npu()
        return tokenizer, model

    def get_dataset_list(self):
        return ["MMLU", "CEval"]
    
    def get_chip_num(self):
        return 2


def main():
    Bloom7BModelTest.create_instance()

if __name__ == "__main__":
    main()
