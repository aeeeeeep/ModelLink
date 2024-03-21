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

    def prepare_environ(self):
        # memory
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        # performance
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['LCCL_ENABLE_FALLBACK'] = "1"
        
    def get_dataset_list(self):
        return ["HumanEval"]


def main():
    CodegeeX26BModelTest.create_instance()

if __name__ == "__main__":
    main()
