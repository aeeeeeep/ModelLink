# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from transformers import AutoTokenizer, AutoModel
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from base import model_test
import shutil
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ModuleNotFoundError:
    pass


class Chatglm26BModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_model(self, hardware_type):
        if hardware_type == "NPU":
            MODEL_FILE = os.path.join(model_test.ATB_SPEED_HOME_PATH, "pytorch/examples/chatglm2_6b/patches/modeling_chatglm_fa.py")
            WEIGHT_DIR_NPU = os.path.join(model_test.ATB_TESTDATA_PATH, "weights", "chatglm2_6b")
            tokenizer = AutoTokenizer.from_pretrained(
                        WEIGHT_DIR_NPU,
                        padding_side="left",
                        trust_remote_code=True,
                    )
            if not self.is_format_nz:
                shutil.copy(MODEL_FILE, os.path.join(WEIGHT_DIR_NPU, "modeling_chatglm.py"))
                model = AutoModel.from_pretrained(WEIGHT_DIR_NPU,
                                                trust_remote_code=True, torch_dtype=torch.half, device='npu')
            else:
                part_model_path = os.path.join(WEIGHT_DIR_NPU, "part_model", str(torch.distributed.get_rank()))
                shutil.copy(MODEL_FILE, os.path.join(part_model_path, "modeling_chatglm.py"))
                model = AutoModel.from_pretrained(part_model_path,
                                                trust_remote_code=True, torch_dtype=torch.half, device='npu')
            model = model.eval()
            model.set_weight()
            return tokenizer, model
        elif hardware_type == "GPU":
            WEIGHT_DIR_GPU = os.environ.get("chatglm2_6b_weight")
            tokenizer = AutoTokenizer.from_pretrained(WEIGHT_DIR_GPU, trust_remote_code=True, padding_side='left')
            model = AutoModel.from_pretrained(WEIGHT_DIR_GPU, trust_remote_code=True).half().cuda()
            model = model.eval()
            return tokenizer, model

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

    def set_fa_tokenizer_params(self):
        self.tokenizer_params = {
            'revision': None,
            'use_fast': False,
            'padding_side': 'left',
            'truncation_side': 'left',
            'trust_remote_code': True
        }


def main():
    Chatglm26BModelTest.create_instance()

if __name__ == "__main__":
    main()
