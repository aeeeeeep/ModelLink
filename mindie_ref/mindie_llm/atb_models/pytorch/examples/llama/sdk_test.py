# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import torch
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher, ParallelLauncher
from atb_speed.common.precision import get_precision_test_cls
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaLM(Launcher):
    """
    单芯推理launcher
    """
    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False,
                                                  padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True,
                                                     torch_dtype=torch.float16).to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer
    

class LlamaLMParallel(ParallelLauncher):
    """
    多芯推理launcher
    """
    def init_model(self):
        tokenizer_path = os.path.join(self.model_path, "tokenizer")
        part_model_path = os.path.join(self.model_pathk, "part_model", str(self.local_rank))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False,
                                                  padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(part_model_path, trust_remote_code=True,
                                                     torch_dtype=torch.float16).to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run Ascend LLM")
    parser.add_argument(
        "--task",
        type=str,
        default='run',
        choices=['run', 'precision'],
        help="Specify the task in which to run the script [--task run|precision]"
    )
    args = parser.parse_args()
    return args


def run_example(launcher):
    # warm-up
    launcher.logger.info("warm-up start...")
    launcher.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->', {"max_new_tokens": 128})
    launcher.logger.info("warm-up success!")
    # inference
    launcher.logger.info("inference start...")
    query_list = [
        "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:"
    ]
    launcher.infer_batch(query_list, {"max_new_tokens": 128})
    launcher.logger.info("inference success!")


def run_precision(launcher):
    precision_tester = get_precision_test_cls()(launcher)
    precision_tester.run()


TASK_MAP = {
    "run": run_example,
    "precision": run_precision
}


def main():
    args = parse_args()
    atb_speed_config.init_config("sdk_config.ini")
    if atb_speed_config.model.device_num > 1:
        launcher = LlamaLMParallel()
    else:
        launcher = LlamaLM()
    TASK_MAP.get(args.task)(launcher)


if __name__ == "__main__":
    main()
