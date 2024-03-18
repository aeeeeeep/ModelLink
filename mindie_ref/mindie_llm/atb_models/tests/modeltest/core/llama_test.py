# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
import shutil
from base import model_test


class LlamaModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        weight_dir = args[12]
        model_name = "llama"
        config_path = os.path.join(weight_dir, "config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            if "max_sequence_length" in config_data:
                model_name = "llama_65b"
            elif "num_hidden_layers" in config_data:
                if config_data["num_hidden_layers"] == 32:
                    model_name = "llama2_7b"
                elif config_data["num_hidden_layers"] == 40:
                    model_name = "llama2_13b"
                elif config_data["num_hidden_layers"] == 80:
                    model_name = "llama2_70b"
        updated_args = args[:3] + (model_name,) + args[4:]
        source_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result", "llama")
        destination_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result", model_name)
        if os.path.exists(destination_dir):
            shutil.rmtree(destination_dir)
        shutil.move(source_dir, destination_dir)
        super().__init__(*updated_args)
        
    def get_chip_num(self):
        return 8
    
    def get_model(self, hardware_type, model_type, data_type):
        pass

    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['INF_NAN_MODE_ENABLE'] = "0"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['LCCL_ENABLE_FALLBACK'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        os.environ['ATB_CONTEXT_WORKSPACE_SIZE'] = "0"

    def get_dataset_list(self):
        return ["BoolQ"]


def main():
    LlamaModelTest.create_instance()

if __name__ == "__main__":
    main()