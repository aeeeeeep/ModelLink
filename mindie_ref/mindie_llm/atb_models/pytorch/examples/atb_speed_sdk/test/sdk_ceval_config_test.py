from transformers import AutoTokenizer, AutoModelForCausalLM

from atb_speed.common.ceval.base import CEvalTest
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher


class BaichuanLM(Launcher):
    def init_model(self):
        """
        模型初始化
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


if __name__ == '__main__':
    atb_speed_config.init_config("config.ini")
    baichuan = BaichuanLM()
    c_t = CEvalTest(baichuan)
    c_t.run_ceval()
