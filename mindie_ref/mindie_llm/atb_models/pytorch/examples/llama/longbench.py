# MIT License

# Copyright (c) 2023 THU-KEG & Zhipu AI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import json
<<<<<<< HEAD
import math
=======
>>>>>>> 838bb9a1877ea3b6d14d8390470d62c3b15376d3
import random
import argparse
from dataclasses import dataclass, field
import torch
import torch_npu
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np


@dataclass
class PredictConfig:
    data: list = field(default_factory=list)
    max_length: int = 0
    max_gen: int = 0
    prompt_format: str = ""
    dataset: str = ""
    model_name: str = ""
    out_path: str = ""


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="yi_6b_200k", choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "yi_6b_200k"])
    parser.add_argument('--model_path', type=str, default="/home/data/yi_6b_200k")
    parser.add_argument('--dataset_path', type=str, default="/home/dataset")
    parser.add_argument('--max_length', type=int, default=15500)
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args()


def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank_ = torch.distributed.get_rank()
    world_size_ = torch.distributed.get_world_size()
    torch_npu.npu.set_device(local_rank_)
    return local_rank_, world_size_


def prepare_environ():
    torch.npu.set_compile_mode(jit_compile=False)
    os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1" 
    os.environ['TASK_QUEUE_ENABLE'] = "1" 
    os.environ['HCCL_BUFFSIZE'] = "110" 
    os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "0" 
    os.environ['ATB_CONTEXT_WORKSPACE_RING'] = "1" 
    os.environ['ATB_USE_TILING_COPY_STREAM'] = "0" 
    os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1" 
    os.environ['LCCL_ENABLE_FALLBACK'] = "1" 
    os.environ['LONG_SEQ_ENABLE'] = "1" 
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048" 
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version not in [104, 220, 221, 222, 223, 224]:
        os.environ['ATB_USE_TILING_COPY_STREAM'] = "1" 


# This is the customized building prompt for chat models
def build_chat(prompt):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name or "yi" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(config):
    for json_obj in tqdm(config.data):
        prompt = config.prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in config.model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if not local_rank:
            print(f"input length: {len(tokenized_prompt)}")
        if len(tokenized_prompt) > config.max_length:
            continue
        if config.dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(prompt)
        if "chatglm3" in model_name:
            inputs = prompt
        else:
            inputs = tokenizer(prompt, truncation=False, return_tensors="pt")
        context_length = inputs.input_ids.shape[-1]
        try:
            if config.dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    inputs.input_ids.npu(),
                    attention_mask=inputs.attention_mask.npu(),
                    max_new_tokens=config.max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    inputs.input_ids.npu(),
                    attention_mask=inputs.attention_mask.npu(),
                    max_new_tokens=config.max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
        except Exception:
            if not local_rank:
                print(f"oom, skip this data!")
            continue
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred)
        torch.npu.empty_cache()
        if not local_rank:
            with open(config.out_path, "a", encoding="utf-8") as result:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, result, ensure_ascii=False)
                result.write('\n')


def load_model_and_tokenizer(path):
    if world_size > 1:
        tokenizer_path = os.path.join(model_path, "tokenizer")
        part_model_path = os.path.join(model_path, "part_model", str(local_rank))
    else: 
        tokenizer_path, part_model_path = path, path
    if "llama2" in model_name:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        tokenizer_ = LlamaTokenizer.from_pretrained(tokenizer_path)
        model_ = LlamaForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.bfloat16).npu()
    elif "longchat" in model_name or "vicuna" in model_name:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        from fastchat.model import load_model
        model_, _ = load_model(
            part_model_path,
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model_ = model_.npu()
        model_ = model_.bfloat16()
        tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
    else:
        tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model_ = AutoModelForCausalLM.from_pretrained(part_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).npu()
    model_ = model_.eval()
    return model_, tokenizer_


def get_result_scores():
    reuslt_dir = os.getenv("RESULT_DIR", "")
    file_path = os.path.join(reuslt_dir, "result.json")
    csv_title = ""
    csv_content = ""
    avg_all = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            result = json.load(f)
            for dataset, res in result.items():
                avg_dataset = []
                csv_title += f"{dataset},"
                for _, score in res.items():
                    if not math.isnan(score):
                        avg_dataset.append(score)
                avg_dataset = round(np.mean(avg_dataset),2)
                avg_all.append(avg_dataset)
                csv_content += f"{avg_dataset},"
    else:
        print("RESULT_DIR is not a real path, please correct.")
    avg_all = round(np.mean(avg_all),2)
    csv_title += "avg"
    csv_content += f"{avg_all}"
    with open(os.path.join(reuslt_dir, "result.csv"), 'w') as f:
        f.write(csv_title)
        f.write("\n")
        f.write(csv_content)
    print(f"longbench avg scores is {avg_all}.")


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path
    dataset_path = args.dataset_path
    max_length = args.max_length

    seed_everything(args.seed)
    prepare_environ()
    # define your model
    local_rank, world_size = setup_model_parallel() if args.world_size > 1 else (0, 1)
    model, tokenizer = load_model_and_tokenizer(model_path)
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    with open("./dataset2prompt.json", "r") as file:
        dataset2prompt = json.load(file)
    with open("./dataset2maxlen.json", "r") as file:
        dataset2maxlen = json.load(file)
    # predict on each dataset
    suffix = "_e" if args.e else ""
    if not os.path.exists(f"pred{suffix}"):
        os.makedirs(f"pred{suffix}")
    for dataset in datasets:
        predict_config = PredictConfig()
        file_path = os.path.join(dataset_path, f"{dataset}{suffix}.jsonl")
        with open(file_path, 'r', encoding="utf-8") as input_file:
            for line in input_file:
                data_line = json.loads(line)
                predict_config.data.append(data_line)
        if not os.path.exists(f"pred{suffix}/{model_name}"):
            os.makedirs(f"pred{suffix}/{model_name}")
        predict_config.max_length = max_length
        predict_config.max_gen = dataset2maxlen[dataset]
        predict_config.prompt_format = dataset2prompt[dataset]
        predict_config.dataset = dataset
        predict_config.model_name = model_name
        predict_config.out_path = f"pred{suffix}/{model_name}/{dataset}.jsonl"
        get_pred(predict_config)
