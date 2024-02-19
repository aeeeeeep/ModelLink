# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
import sys
sys.path.append(os.path.join(ATB_SPEED_HOME_PATH, "../.."))
from atb_llm.runner import ModelRunner
from base import model_test
from transformers.generation.utils import GenerationConfig

MODEL_FILE = os.path.join(
    model_test.ATB_SPEED_HOME_PATH,
    "pytorch/examples/baichuan2/13b/modeling_baichuan_ascend.py",
)
WEIGHT_DIR = os.path.join(model_test.ATB_TESTDATA_PATH, "weights", "baichuan2_13b")


class Baichuan213BModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def remove_part_of_generation_config(self, generation_config):
        ori_gen = GenerationConfig()
        diff_dict = generation_config.to_diff_dict()
        print(diff_dict)
        for key in diff_dict:
            if key.endswith("_id"):
                continue
            ori_value = getattr(ori_gen, key, None)
            if ori_value is not None:
                setattr(generation_config, key, getattr(ori_gen, key))
                print(f"replace {key}")
        return generation_config

    def get_model(self, hardware_type, model_type, data_type):
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))

        dtype = model_test.dtype_map.get(data_type, "fp16")
        model = ModelRunner(
            self.weight_dir, rank=rank, world_size=world_size, dtype=dtype, quantize=None
        )
        tokenizer = model.tokenizer
        model.load_weights()
        return tokenizer, model

    def get_dataset_list(self):
        return ["GSM8K", "MMLU", "CEval"]


def main():
    Baichuan213BModelTest.create_instance()


if __name__ == "__main__":
    main()
