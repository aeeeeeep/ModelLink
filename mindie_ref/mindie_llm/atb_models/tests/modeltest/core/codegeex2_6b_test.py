# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import shutil
from transformers import AutoTokenizer, AutoModel
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ModuleNotFoundError:
    pass
from base import model_test


class CodegeeX26BModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_dataset_list(self):
        return ["GSM8K", "MMLU", "CEval"]


def main():
    CodegeeX26BModelTest.create_instance()

if __name__ == "__main__":
    main()
