# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from base import model_test


class LlamaModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    
    def get_model(self, hardware_type, model_type, data_type):
        pass

    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['LCCL_ENABLE_FALLBACK'] = "1"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"

    def get_dataset_list(self):
        return ["CEval", "BoolQ"]


def main():
    LlamaModelTest.create_instance()

if __name__ == "__main__":
    main()