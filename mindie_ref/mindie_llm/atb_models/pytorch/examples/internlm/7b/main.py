import argparse
import os

from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import ParallelLauncher, Launcher
from atb_speed.common.performance.base import PerformanceTest
from atb_speed.common.precision import get_precision_test_cls
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Adapting LLM on Ascend")
    parser.add_argument(
        "--task",
        type=str,
        default='inference',
        choices=['inference', 'precision', 'performance'],
        help="Specify the task in which to run the script"
    )
    args = parser.parse_args()
    return args


class BaichuanLMParallel(ParallelLauncher):
    """
    多卡推理launcher
    """

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False,
                                                  padding_side="left")

        part_model_path = os.path.join(self.model_path, 'part_model', str(self.local_rank))
        model = AutoModelForCausalLM.from_pretrained(part_model_path, trust_remote_code=True)
        model = model.half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


class BaichuanLM(Launcher):
    """
    单卡推理launcher
    """

    def init_model(self):
        """
        模型初始化
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False,
                                                  padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


def demo_ceval(launcher: Launcher):
    """

    :param launcher:
    :return:
    """
    c_t = get_precision_test_cls()(launcher)
    c_t.run()


def demo_perf(launcher: Launcher):
    """

    :param launcher:
    :return:
    """
    performance_test = PerformanceTest(launcher)
    performance_test.warm_up()
    performance_test.run_test()


def demo_inference(launcher: Launcher):
    """

    :param launcher:
    :return:
    """
    launcher.logger.info("---------------warm-up---------------")
    launcher.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->',
                   {"max_new_tokens": 64, "do_sample": False, "repetition_penalty": 1.1})

    launcher.logger.info("---------------inference---------------")
    launcher.infer('登鹳雀楼->王之涣\n夜雨寄北->',
                   {"max_new_tokens": 64, "do_sample": False, "repetition_penalty": 1.1})

    launcher.logger.info("---------------2k---------------")
    launcher.infer_test(1, 2048, 64)

    launcher.logger.info("---------------batch---------------")
    query_list = ["谷歌公司的CEO是",
                  '登鹳雀楼->王之涣\n夜雨寄北->',
                  '苹果公司的CEO是',
                  '华为公司的CEO是',
                  '微软公司的CEO是']
    launcher.infer_batch(query_list, {"max_new_tokens": 64, "do_sample": False, "repetition_penalty": 1.1})


TASK_MAP = {
    "inference": demo_inference,
    "precision": demo_ceval,
    "performance": demo_perf
}


def main():
    args = parse_args()
    atb_speed_config.init_config("config.ini")
    if atb_speed_config.model.device_num > 1:
        launcher = BaichuanLMParallel()
    else:
        launcher = BaichuanLM()
    TASK_MAP.get(args.task)(launcher)


if __name__ == "__main__":
    main()
