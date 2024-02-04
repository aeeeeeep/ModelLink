import argparse
import os
import json
import sys
import time
import itertools
import torch

from collections import defaultdict

import numpy as np
import eval
from atb_speed.common.ceval.base import CEvalTest
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import ParallelLauncher, Launcher
from atb_speed.common.performance.base import PerformanceTest
from transformers import AutoTokenizer, AutoModelForCausalLM

from modeling_llama_fa_rope import LlamaForCausalLM

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Adapting Codellama-34B on Ascend")
    parser.add_argument(
        "--task",
        type=str,
        default='inference',
        choices=['inference', 'ceval', 'performance', 'human_eval'],
        help="Specify the task in which to run the script"
    )
    args = parser.parse_args()
    return args


class CodeLlama34BParallel(ParallelLauncher):
    """
    多卡推理launcher
    """

    def infer(self, query):
        """
        推理代码
        :param query:
        :return:
        """

        system = DEFAULT_SYSTEM_PROMPT
        prompts = self.get_prompt(query, system)
        gen_kwargs = dict(
            max_new_tokens=2048,
            top_p=0.90,
            top_k=50,
            temperature=0.1,
            bos_token_id=1,
            eos_token_id=2,
            do_sample=False,
            num_beams=1,
        )
        inputs = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False)

        generation_kwargs = dict(
            inputs=inputs["input_ids"].npu(),
            attention_mask=inputs['attention_mask'].npu(),
            **gen_kwargs
        )
        print(f"问:{query}")
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **generation_kwargs
            )
        output = self.tokenizer.decode(outputs[0].cpu().numpy().tolist())
        print(f'CodeLlama34b-答:\n')
        print(output)
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"cost {time_cost}s")
        return output

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.model_path, 'tokenizer'), trust_remote_code=True, use_fast=False,
                                                  padding_side="left")
        part_model_path = os.path.join(self.model_path, 'part_model', str(self.local_rank))
        model = LlamaForCausalLM.from_pretrained(part_model_path, trust_remote_code=True)
        model = model.half().to(self._device)
        model.eval()
        return model, tokenizer

    def get_prompt(self, message: str,
                   system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    def infer_human_eval(self, query):
        """
        推理代码
        :param query:问题
        :param model_params:
        :return:
        """
        prompt = self.get_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        inputs = inputs.to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=2048,
            top_p=0.90,
            top_k=50,
            bos_token_id=1,
            eos_token_id=2,
            temperature=0.1,
            do_sample=True,
            num_beams=1,
        )
        with torch.no_grad():
            start_time = time.time()
            pred = self.model.generate(**inputs, **gen_kwargs)
            end_time = time.time()
            time_cost = end_time - start_time
        output = self.tokenizer.decode(pred[0].cpu().numpy().tolist())
        new_tokens = len(pred[0]) - len(inputs.input_ids[0])
        self.logger.info(f"generate {new_tokens} new tokens，({new_tokens / time_cost:.2f} tokens/s")
        print(f'output = {output}')
        resp = output.replace(query, "")
        start = resp.find("```python")
        have_python = "```python"
        if start < 0:
            start = resp.find("```")
            have_python = "```"
        if start < 0:
            start = resp.find("[PYTHON]")
            have_python = "[PYTHON]"
        if start >= 0:
            end = -1
            if have_python == "```python":
                start = start + len("```python")
                end = resp.find("```", start)
            elif have_python == "```":
                start = start + len("```")
                end = resp.find("```", start)
            elif have_python == "[PYTHON]":
                start = start + len("[PYTHON]")
                end = resp.find("[/PYTHON]", start)
            if end >= 0:
                resp = resp[start:end]
        print(f'resp = {resp}')
        return resp


class CodeLlama34B(Launcher):
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
        model = LlamaForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().to(self._device)
        model.eval()
        return model, tokenizer


def demo_ceval(launcher: Launcher):
    """
    ceval测试
    :param launcher:Launcher
    :return:
    """
    c_t = CEvalTest(launcher)
    c_t.run_ceval()


def demo_perf(launcher: Launcher):
    """
    性能测试

    :param launcher:Launcher
    """
    performance_test = PerformanceTest(launcher)
    performance_test.warm_up()
    performance_test.run_test()


def demo_inference(launcher: Launcher):
    """
    推理
    :param launcher:Launcher
    """
    launcher.logger.info("---------------warm-up---------------")
    launcher.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->',
                   {"max_new_tokens":128, "do_sample":False, "repetition_penalty":1.0})

    launcher.infer('Please use Java to implement a binary search algorithm.')

    # launcher.logger.info("---------------inference---------------")
    launcher.infer('登鹳雀楼->王之涣\n夜雨寄北->', {"max_new_tokens":64, "do_sample":False, "repetition_penalty":1.0})

    # launcher.logger.info("---------------2k---------------")
    launcher.infer_test(1, 2048, 64)

    # launcher.logger.info("---------------batch---------------")
    query_list = ["谷歌公司的CEO是",
                  '登鹳雀楼->王之涣\n夜雨寄北->',
                  '苹果公司的CEO是',
                  '华为公司的CEO是',
                  '微软公司的CEO是']
    launcher.infer_batch(query_list, {"max_new_tokens":64, "do_sample":False, "repetition_penalty":1.0})


def compute(datas, max_k, launcher):
    """
    数据集评测主要入口函数
    Args:
        datas: 数据集
        max_k: pass_at_k中的k值
    Returns: pass_at_k
    """
    results = defaultdict(list)
    for task_id, doc in enumerate(datas):
        i = 0
        while (i < max_k):
            start = int(time.time())
            # 获取数据集的prompt
            prompt = eval.get_prompt(doc)
            prompt = eval.build_prompt(prompt)
            # 获取模型推理结果
            candidate = launcher.infer_human_eval(prompt)
            # 获取数据集测试用例
            reference = eval.get_reference(doc)
            # 运行测试用例，获取评测结果
            evaluateDict = eval.evaluate(task_id, candidate, reference)
            results[evaluateDict["task_id"]].append((evaluateDict["completion_id"], evaluateDict))
            end = int(time.time())
            print(f"### candidate: \n{evaluateDict['candidate']}")
            print(f"### task_id: {str(task_id)} ### i: {str(i)} ### test_result: {evaluateDict['result']} "
                  f"### time cost: {str(end - start)}s")
            time.sleep(5)
            i += 1
    # 计算pass_at_k
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    ks = [1]
    pass_at_k = {f"pass@{k}": eval.estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    return pass_at_k, results


def demo_human_eval(launcher: Launcher):
    """
    humaneval测试
    :param launcher:Launcher
    """
    datas = []
    dataset_path = r'human-eval-v2-20210705.jsonl'
    print("testset path: " + dataset_path)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(str(line))
        datas.append(data)
    pass_at_k, results = compute(datas, 1, launcher)
    print("pass_at_k = " + str(pass_at_k))
    print("test end")


TASK_MAP = {
    "inference": demo_inference,
    "ceval": demo_ceval,
    "performance": demo_perf,
    "human_eval": demo_human_eval
}


def main():
    args = parse_args()
    atb_speed_config.init_config("config.ini")
    if atb_speed_config.model.device_num > 1:
        launcher = CodeLlama34BParallel()
    else:
        launcher = CodeLlama34B()
    TASK_MAP.get(args.task)(launcher)


if __name__ == "__main__":
    main()
