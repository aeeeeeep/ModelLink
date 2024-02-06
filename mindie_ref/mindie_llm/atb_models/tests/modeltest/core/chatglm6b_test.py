# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import shutil
from transformers import AutoTokenizer, AutoModel
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ModuleNotFoundError:
    pass
from base import model_precision_test


class Chatglm6BModelTest(model_precision_test.ModelPrecisionTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_model(self, hardware_type):
        super().get_model(hardware_type)

        if hardware_type == "NPU":
            MODEL_FILE = os.path.join(model_precision_test.ATB_SPEED_HOME_PATH, "pytorch/examples/chatglm6b/modeling_chatglm.py")
            WEIGHT_DIR_NPU = os.path.join(model_precision_test.ATB_TESTDATA_PATH, "weights", "chatglm6b")
            tokenizer = AutoTokenizer.from_pretrained(
                        WEIGHT_DIR_NPU,
                        padding_side="left",
                        trust_remote_code=True,
                    )
            
            torch.manual_seed(1)
            shutil.copy(MODEL_FILE, os.path.join(WEIGHT_DIR_NPU, "modeling_chatglm.py"))
            model = AutoModel.from_pretrained(WEIGHT_DIR_NPU,
                                            trust_remote_code=True).half().npu()
            torch.npu.set_compile_mode(jit_compile=False)

            model = model.eval()

            # 确认配置
            ENABLE_QUANT = os.environ.get("ENABLE_QUANT", "0") == "1"
            is_format_nz = self.__get_is_format_nz()
            if ENABLE_QUANT:
                QUANT_WEIGHT_PATH = os.environ.get("QUANT_WEIGHT_PATH")

            # 浮点模型适配
            if is_format_nz:
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)

            return tokenizer, model
        elif hardware_type == "GPU":
            WEIGHT_DIR_GPU = os.environ.get("chatglm6b_weight")
            tokenizer = AutoTokenizer.from_pretrained(WEIGHT_DIR_GPU, trust_remote_code=True, padding_side='left')
            model = AutoModel.from_pretrained(WEIGHT_DIR_GPU, trust_remote_code=True).half().cuda()
            model = model.eval()
            return tokenizer, model

    def get_dataset_list(self):
        return ["GSM8K", "MMLU", "CEval"]

    def __get_is_format_nz(self):
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version in [200, 201, 202, 203]:
            return True
        elif soc_version in [220, 221, 222, 223, 224]:
            return False
        else:
            raise NotImplementedError


def main():
    Chatglm6BModelTest.create_instance()

if __name__ == "__main__":
    main()
