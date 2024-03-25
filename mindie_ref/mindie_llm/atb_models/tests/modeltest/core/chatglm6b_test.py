# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from base import model_test


class Chatglm6BModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_model(self, hardware_type):
        pass

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
        return ["BoolQ", "CEval"]


def main():
    Chatglm6BModelTest.create_instance()

if __name__ == "__main__":
    main()

