#coding=utf-8

import argparse
import os

from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import ParallelLauncher, Launcher
from atb_speed.common.performance.base import PerformanceTest
from atb_speed.common.precision import get_precision_test_cls
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Adapting LLM on Ascend")
    parser.add_argument(
        "--task",
        type=str,
        default='inference',
        choices=['inference', 'precision', 'performance'],
        help="Specify the task in which to run the script"
    )
    parser.add_argument(
        "--is_quant",
        type=str,
        default="0",
        choices=['0', '1'],
        help="Specify the quant model [1] or float model [0]"
    )
    args = parser.parse_args()
    return args


class BaichuanConfig(PretrainedConfig):
    model_type = "baichuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=125696,
            hidden_size=5120,
            intermediate_size=13696,
            num_hidden_layers=40,
            num_attention_heads=40,
            hidden_act="silu",
            model_max_length=4096,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            gradient_checkpointing=False,
            z_loss_weight=0,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.z_loss_weight = z_loss_weight
        self.gradient_checkpointing = (gradient_checkpointing,)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


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
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id  # set as the token
        if tokenizer.pad_token_id == 64000:
            tokenizer.pad_token_id = 0  # for baichuan model (need fix)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
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
        print(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False,
                                                  padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().to(self._device)
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id  # set as the token
        if tokenizer.pad_token_id == 64000:
            tokenizer.pad_token_id = 0  # for baichuan model (need fix)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
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
    launcher.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->', {"max_new_tokens": 128})

    launcher.logger.info("---------------inference---------------")
    launcher.infer('登鹳雀楼->王之涣\n夜雨寄北->', {"max_new_tokens": 128})

    # launcher.logger.info("---------------2k---------------")
    # launcher.infer_test(1, 2048, 32)

    launcher.logger.info("---------------batch---------------")
    query_list = ['登鹳雀楼->王之涣\n夜雨寄北->',
                  '苹果公司的CEO是',
                  '华为公司的CEO是',
                  '微软公司的CEO是']
    launcher.infer_batch(query_list, {"max_new_tokens": 64})


TASK_MAP = {
    "inference": demo_inference,
    "precision": demo_ceval,
    "performance": demo_perf
}


def main():
    args = parse_args()
    atb_speed_config.init_config("config.ini")
    if atb_speed_config.model.device_num > 1:
        if args.is_quant == '1':
            launcher = BaichuanLMParallelQuant()
        else:
            launcher = BaichuanLMParallel()
    else:
        if args.is_quant == '1':
            launcher = BaichuanLMQuant()
        else:
            launcher = BaichuanLM()
    TASK_MAP.get(args.task)(launcher)


if __name__ == "__main__":
    main()
