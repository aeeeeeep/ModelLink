import argparse
import os
import torch
import psutil

from atb_speed.common.ceval.base import CEvalTest
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import ParallelLauncher, Launcher
from atb_speed.common.performance.base import PerformanceTest
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from modeling_internlm_quant_ascend_optimization import InternLMForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Adapting InternLM-20B on Ascend")
    parser.add_argument(
        "--task",
        type=str,
        default='inference',
        choices=['inference', 'ceval', 'performance'],
        help="Specify the task in which to run the script"
    )
    args = parser.parse_args()
    return args


class InternLMLMParallel(ParallelLauncher):

    def _get_cpu_info(self, numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
        cpu_idx_tbl = dict()
        numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
        with os.popen(f"lscpu") as f:
            cpu_info = f.read().strip().split("\n")
        for _ in cpu_info:
            line = ''.join(_.split())
            if any(line.startswith(word) for word in numa_keywords):
                split_info = line.split("：")
                cpu_id_ranges = split_info[-1].split(",")
                ranges = list()
                for range_str in cpu_id_ranges:
                    endpoints = range_str.split("-")
                    print(f'endpoints = {endpoints}')
                    if len(endpoints) != 2:
                        raise Exception("lscpu command output error, please check !")

                    ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]

                numa_id = int(split_info[0].replace(keyword1, '').replace(keyword2, ''))
                cpu_idx_tbl[numa_id] = ranges
        return cpu_idx_tbl

    def bind_cpus(self, world_size, local_rank, ratio=0.5):
        # docker中npu亲和性查询会失效，默认是用numa node0
        numa_id = 0
        cpu_idx_tbl = self._get_cpu_info([numa_id], 'NUMA节点', 'CPU')
        print(f'cpu_idx_tbl = {cpu_idx_tbl}')
        all_cpus = cpu_idx_tbl[numa_id]

        cpu_nums = len(all_cpus)
        # 计算给该共享numa的npu分配的核的个数
        cpu_num_per_device = int(cpu_nums * ratio // world_size)

        # 给该npu分配要绑定的cpu id
        binding_cpus = [all_cpus[_] for _ in
                        range(local_rank * cpu_num_per_device, (local_rank + 1) * cpu_num_per_device)]

        # cpu bind
        p = psutil.Process()
        p.cpu_affinity(binding_cpus)
        new_affinity = p.cpu_affinity()

    """
    多卡推理launcher
    """
    def init_model(self):
        tokenizer_path = self.model_path + '/tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side='left',
                                                  use_fast=False)

        part_model_path = self.model_path + '/part_model/' + str(self.local_rank) + '/'
        config = AutoConfig.from_pretrained(part_model_path, trust_remote_code=True)  # 加载
        config._name_or_path = part_model_path
        model = InternLMForCausalLM(config).half().npu()
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


class InternLMLM(Launcher):
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
    c_t = CEvalTest(launcher)
    c_t.run_ceval()


def demo_perf(launcher: Launcher):
    """

    :param launcher:
    :return:
    """
    performance_test = PerformanceTest(launcher)
    performance_test.warm_up()
    performance_test.run_test()
    # performance_test.run_single_test(3524, 512)


def demo_inference(launcher: Launcher):
    """

    :param launcher:
    :return:
    """
    launcher.logger.info("---------------warm-up---------------")
    launcher.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->',
                   {"max_new_tokens": 16, "do_sample": False, "repetition_penalty": 1.0})

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
    "ceval": demo_ceval,
    "performance": demo_perf
}


def main():
    args = parse_args()
    atb_speed_config.init_config("config.ini")
    if atb_speed_config.model.device_num > 1:
        launcher = InternLMLMParallel()
    else:
        launcher = InternLMLM()
    TASK_MAP.get(args.task)(launcher)


if __name__ == "__main__":
    main()
