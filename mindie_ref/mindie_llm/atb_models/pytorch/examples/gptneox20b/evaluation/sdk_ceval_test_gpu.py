import os
import time

import torch

from atb_speed.common.ceval.base import CEvalTest
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher.gpu import Launcher
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from configuration_gpt_neox import GPTNeoXConfig


class GPTNeox(Launcher):
    def init_model(self):
        """
        模型初始化
        :return:
        """
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        config = GPTNeoXConfig.from_pretrained(self.model_path)
        config.is_decoder = True
        print(f"Done load model config in cpu cost: {time.time() - start_time}s, ", config)
        model = GPTNeoXForCausalLM.from_pretrained(self.model_path, config=config)
        print(f"Done load model in cpu, cost: {time.time() - start_time}s")

        model.gradient_checkpointing_disable()
        device_id = os.getenv('CUDA_VISIBLE_DEVICES')
        model.half().to(torch.device(f"cuda:{device_id}"))
        model.eval()

        return model, tokenizer


if __name__ == '__main__':
    atb_speed_config.init_config("config.ini")
    GPTNeox = GPTNeox("0")
    c_t = CEvalTest(GPTNeox)
    c_t.run_ceval()
