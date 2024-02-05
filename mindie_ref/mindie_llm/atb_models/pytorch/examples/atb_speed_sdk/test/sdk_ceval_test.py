import argparse
import os

from atb_speed.common.ceval.base import CEvalTest
from atb_speed.common.launcher import Launcher
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--model_path",
        default="/data/model",
        help="Location of Model weights, which contains model folders",
    )
    args = parser.parse_args()
    baichuan = BaichuanLM("1", args.model_path)
    c_t = CEvalTest(baichuan, os.path.realpath(os.path.dirname(__file__)))
    c_t.run_ceval()
