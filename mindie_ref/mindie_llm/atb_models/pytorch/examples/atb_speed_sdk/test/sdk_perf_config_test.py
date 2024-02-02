from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher
from atb_speed.common.performance.base import PerformanceTest
from transformers import AutoTokenizer, AutoModelForCausalLM


class Baichuan2_7B(Launcher):
    """
    Baichuan2_7B_NPU
    """

    def init_model(self):
        """
        模型初始化
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


if __name__ == '__main__':
    atb_speed_config.init_config("config.ini")
    performance_test = PerformanceTest(Baichuan2_7B("0"))
    performance_test.warm_up()
    performance_test.run_test()
