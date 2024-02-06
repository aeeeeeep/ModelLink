# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from base import model_test
import os
import shutil
import torch
import torch_npu
 
MODEL_FILE = os.path.join(model_test.ATB_SPEED_HOME_PATH, "pytorch/examples/baichuan2/13b/modeling_baichuan_ascend.py")
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
    
    def get_model(self, hardware_type):
        super().get_model()
 
        tokenizer = AutoTokenizer.from_pretrained(
                    WEIGHT_DIR,
                    padding_side="left",
                    use_fast=False,
                    trust_remote_code=True,
                )
 
        torch.manual_seed(1)
        shutil.copy(MODEL_FILE, os.path.join(WEIGHT_DIR, "modeling_baichuan_ascend.py"))
        model = AutoModelForCausalLM.from_pretrained(WEIGHT_DIR,
                                                    trust_remote_code=True).half().npu()
        torch.npu.set_compile_mode(jit_compile=False)
        model = model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
 
        return tokenizer, model

    def get_dataset_list(self):
        return ["GSM8K", "MMLU", "CEval"]
 
def main():
    Baichuan213BModelTest.create_instance()
 
if __name__ == "__main__":
    main()