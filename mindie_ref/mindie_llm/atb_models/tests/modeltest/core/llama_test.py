# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import ast
ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
import sys
sys.path.append(os.path.join(ATB_SPEED_HOME_PATH, "../.."))
import torch
from base import model_test
from atb_llm.runner import ModelRunner


class LlamaModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        self.weight_dir = args[12]
        self.use_refactor = args[13]
        self.max_position_embeddings = args[14] if args[14] != -1 else None
        super().__init__(*args)

    def get_chip_num(self):
        return 8
    
    def get_model(self, hardware_type, model_type, data_type):
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))

        if model_type == "fa":
            model = ModelRunner(self.weight_dir, rank=rank, world_size=world_size, quantize=None, dtype=torch.float16,
                            is_flash_causal_lm=False, use_refactor=self.use_refactor)
            tokenizer = model.tokenizer
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})   
        else:
            dtype = model_test.dtype_map[data_type] if data_type in model_test.dtype_map else model_test.dtype_map["fp16"]
            model = ModelRunner(
                self.weight_dir, rank=rank, world_size=world_size, dtype=dtype,
                max_position_embeddings=self.max_position_embeddings, quantize=None, use_refactor=self.use_refactor
            )
            tokenizer = model.tokenizer
        model.load_weights()    
        return tokenizer, model

    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['INF_NAN_MODE_ENABLE'] = "0"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['LCCL_ENABLE_FALLBACK'] = "1"

    def get_dataset_list(self):
        return ["GSM8K", "MMLU", "TruthfulQA"]


def main():
    LlamaModelTest.create_instance()

if __name__ == "__main__":
    main()