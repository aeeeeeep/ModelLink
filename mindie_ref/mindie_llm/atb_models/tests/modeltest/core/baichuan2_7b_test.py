# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from base import model_test


class Baichuan27BModelTest(model_test.ModelTest):
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

    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['LCCL_ENABLE_FALLBACK'] = "1"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
        os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:2048'
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = '1'

    def get_dataset_list(self):
        return ["BoolQ", "CEval"]
    
    def set_fa_tokenizer_params(self):
        self.tokenizer_params = {
            'revision': None,
            'use_fast': False,
            'padding_side': 'left',
            'truncation_side': 'left',
            'trust_remote_code': True
        }


def main():
    Baichuan27BModelTest.create_instance()


if __name__ == "__main__":
    main()
