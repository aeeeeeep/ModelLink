from common.launcher import ParallelLauncher
from common.npu_perform import PerformanceTestNPU
from modeling_internlm_quant_ascend_optimization import InternLMForCausalLM
from transformers import AutoTokenizer
from configuration_internlm import InternLMConfig

import argparse


class Internlm20bNpu(ParallelLauncher):
    """
    InternLM_20B_NPU
    """
    def init_model(self):
        tokenizer_path = self.model_path + '/tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, local_files_only=True, padding_side='left', use_fast=False)

        part_model_path = self.model_path + '/part_model/' + str(self.local_rank) + '/'
        config = InternLMConfig.from_pretrained(part_model_path, trust_remote_code=True) # 加载
        model = InternLMForCausalLM(config).half().npu()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="InternLM_20B_NPU")
    parser.add_argument(
        "--model_path",
        default='./',
        help="模型路径",
    )
    args = parser.parse_args()
    performance_test = PerformanceTestNPU(Internlm20bNpu("0,1", args.model_path))
    performance_test.warm_up()
    performance_test.run_test()
