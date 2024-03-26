import argparse
import os
import torch
import torch_npu
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import ParallelLauncher, Launcher
from atb_speed.common.performance.base import PerformanceTest
from atb_speed.common.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Adapting LLM on Ascend")
    parser.add_argument(
        "--task",
        type=str,
        default='performance',
        choices=['performance'],
        help="Specify the task in which to run the script"
    )
    args = parser.parse_args()
    return args


class MiniGPT4LM(Launcher):
    """
    单卡推理launcher
    """

    def init_model(self):
        print(f"self.model_path : {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False,
                                                  padding_side='left')
        # config = AutoConfig.from_pretrained(self.model_path)
        # padding
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().npu().eval()
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer


def demo_perf(launcher: Launcher):
    """
    :param launcher:
    :return:
    """
    performance_test = PerformanceTest(launcher)
    performance_test.warm_up()
    performance_test.run_test()


def main():
    args = parse_args()
    print(args)
    atb_speed_config.init_config("config.ini")
    launcher = MiniGPT4LM()
    Timer.sync = torch.npu.synchronize
    Timer.reset()
    demo_perf(launcher)
    print(f"{Timer.timeit_res.first_token_delay=}")
    print(f"{Timer.timeit_res.next_token_avg_delay=}")


if __name__ == "__main__":
    main()
