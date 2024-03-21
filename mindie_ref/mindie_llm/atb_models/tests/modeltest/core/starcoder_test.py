# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from base import model_test


class StarcoderModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_chip_num(self):
        return 8
    
    def get_model(self, hardware_type, model_type, data_type):
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))

        if model_type == "fa":
            model = ModelRunner(self.weight_dir, rank=rank, world_size=world_size, quantize=None, dtype=torch.float16,
                            is_flash_causal_lm=False, use_refactor=False)
            tokenizer = model.tokenizer
        else:
            dtype = model_test.dtype_map[data_type] if data_type in model_test.dtype_map else model_test.dtype_map["fp16"]
            model = ModelRunner(
                self.weight_dir, rank=rank, world_size=world_size, dtype=dtype,
                max_position_embeddings=self.max_position_embeddings, quantize=None, use_refactor=False
            )
            tokenizer = model.tokenizer
        model.load_weights()    
        return tokenizer, model

    def prepare_environ(self):
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1" 
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['LCCL_ENABLE_FALLBACK'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
    
    def get_dataset_list(self):
        return ["HumanEval"]


def main():
    StarcoderModelTest.create_instance()

if __name__ == "__main__":
    main()