# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
import shutil
from base import model_test


class LlamaModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        weight_dir = args[12]
        model_name = "qwen"
        config_path = os.path.join(weight_dir, "config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            if "num_hidden_layers" in config_data:
                if config_data["num_hidden_layers"] == 40:
                    model_name = "qwen_14b"
                elif config_data["num_hidden_layers"] == 80:
                    model_name = "qwen_72b"
        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)
    
    def get_model(self, hardware_type, model_type, data_type):
        pass

    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['LCCL_ENABLE_FALLBACK'] = "1"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
        os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:2048'
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = '1'

    def get_dataset_list(self):
        return ["CEval", "BoolQ"]


def main():
    LlamaModelTest.create_instance()

if __name__ == "__main__":
    main()