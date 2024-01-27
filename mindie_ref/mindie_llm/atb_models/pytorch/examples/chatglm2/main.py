# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2031. All rights reserved

import ast
import argparse
import glob
import math
import os
import platform
import shutil
import stat
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import transformers
from transformers.utils import check_min_version
import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu


def override_topp_and_topk():
    # 修改transformers的TopKLogitsWarper
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        filter_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(indices_to_remove, filter_value)
        return scores

    transformers.generation.TopKLogitsWarper.__call__ = __call__

    # 修改transformers的TopPLogitsWarper
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False, stable=True)
        cumulative_probs = sorted_logits.softmax(
            dim=-1).cpu().float().cumsum(dim=-1).to(sorted_logits.device).to(sorted_logits.dtype)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        filter_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(indices_to_remove, filter_value)
        return scores

    transformers.generation.TopPLogitsWarper.__call__ = __call__


def parse_args():
    parser = argparse.ArgumentParser(description="Adapting ChatGLM2-6B on Ascend")
    parser.add_argument(
        "--mode",
        type=str,
        default='precision_dataset',
        choices=['precision_single', 'precision_dataset', 'performance', 'cli_demo'],
        help="Specify the mode in which to run the script"
    )
    parser.add_argument("--model_path", type=str, required=True, help="The path to model weights")
    parser.add_argument("--tp_size", type=int, default=1, help="Whether test model in parallel")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument(
        "--model_file", 
        type=str, 
        default="patches/models/modeling_chatglm_fa.py",
        help="The implementation of model"
    )
    parser.add_argument(
        "--ceval_dataset",
        type=str,
        default='',
        help="The path to ceval dataset"
    )
    
    parser.add_argument(
        "--set_case_pair",
        type=int,
        default=0,
        help="set specified case_pair if 1",
    )
    parser.add_argument(
        "--seqlen_in_range",
        default=[5, 10],
        help="input seqlen ranges from 2^5 to 2^10",
    )
    parser.add_argument(
        "--seqlen_out_range",
        default=[5, 10],
        help="output seqlen ranges from 2^5 to 2^10",
    )
    parser.add_argument(
        "--seqlen_in_pair",
        default=[256, 512, 1024],
        help="specified case",
    )
    parser.add_argument(
        "--seqlen_out_pair",
        default=[64, 128, 256],
        help="specified case",
    )
    parser.add_argument(
        "--performance_output_file",
        type=str,
        default='performance.csv',
        help="file name of performance test"
    )
    parser.add_argument(
        "--print_response",
        action="store_true",
        help="print response during performance test"
    )

    args = parser.parse_args()

    return args


def get_is_format_nz():
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [200, 201, 202, 203]:
        return True
    elif soc_version in [220, 221, 222, 223, 224]:
        return False
    else:
        raise NotImplementedError


def padding_zeros(x):
    zeros = torch.zeros(x.shape)
    result = torch.cat(
        (x.unsqueeze(1), zeros.unsqueeze(1)), dim=1).view(-1)
    return result


def check_lists(arg):
    if isinstance(arg, list):
        return arg
    return [int(i) for i in arg.split(',')]

    
def get_model(args):
   
    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.tp_size > 1:
        torch.distributed.init_process_group("hccl")
        local_rank = torch.distributed.get_rank()
        torch_npu.npu.set_device(args.device + local_rank)
        torch.manual_seed(1)
        part_model_path = os.path.join(args.model_path, "tensor_parallel/part_model", str(local_rank))
        shutil.copy(args.model_file, os.path.join(part_model_path, "modeling_chatglm.py"))
        model = AutoModel.from_pretrained(part_model_path,
                                          trust_remote_code=True, torch_dtype=torch.half, device='npu')
    else:
        local_rank = 0
        torch.npu.set_device(args.device)
        torch.manual_seed(1)
        shutil.copy(args.model_file, os.path.join(args.model_path, "modeling_chatglm.py"))
        model = AutoModel.from_pretrained(args.model_path,
                                          trust_remote_code=True, torch_dtype=torch.half, device='npu')
    
    # 使用二进制优化，消除动态shape的编译问题
    torch.npu.set_compile_mode(jit_compile=False)

    # 推理模式
    model = model.eval()

    # 确认配置
    ENABLE_QUANT = os.environ.get("ENABLE_QUANT", "0") == "1"
    is_format_nz = get_is_format_nz()
    if ENABLE_QUANT:
        QUANT_WEIGHT_PATH = os.environ.get("QUANT_WEIGHT_PATH")

    # 浮点模型适配
    if is_format_nz:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)

    model.set_weight()

    return tokenizer, model


def build_prompt(text):
    return f"[Round {1}]\n\n问：{text}\n\n答："


def precision(args, tokenizer, model):
    # 基于开源代码略作修改，详情参见
    # https://github.com/THUDM/ChatGLM2-6B/blob/main/evaluation/evaluate_ceval.py

    local_rank = 0 if not args.tp_size > 1 else torch.distributed.get_rank()

    if args.mode == 'precision_single':
        texts = ['新冠病毒刚刚爆发时检测病患并及时隔离的措施属于____\nA. 注射疫苗\nB. 控制传染源\nC. 切断传播途径\nD. 保护易感人群'] * args.batch
        queries = [build_prompt(query) for query in texts]
        inputs = tokenizer(queries, padding="max_length", return_tensors="pt",
                            truncation=True, max_length=1024).to('npu')
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
        if local_rank == 0:
            print("Question:", end="\n\n")
            for text in texts:
                print(text.replace('\n', '\n\n'), end="\n\n")
            print("Answer:", end="\n\n")
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                response = tokenizer.decode(output)
                print(response)
            print('\n')
        return

    choices = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]
    extraction_prompt = '综上所述，ABCD中正确的选项是：'

    accuracy_dict, count_dict = {}, {}
    json_file = "precison_result.json"
    with torch.no_grad():
        for entry in tqdm(glob.glob((Path(args.ceval_dataset) / "val/**/*.jsonl").as_posix(),
                                    recursive=True)):
            dataset = []
            with open(entry, encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line))
            correct = 0
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch)
            for batch in tqdm(dataloader):
                texts = batch["inputs_pretokenized"]
                queries = [build_prompt(query) for query in texts]
                inputs = tokenizer(queries, padding=True, return_tensors="pt",
                                   truncation=True, max_length=2048).to('npu')
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
                intermediate_outputs = [
                    tokenizer.decode(output)
                    for output in outputs[:, inputs["input_ids"].size(1):].tolist()
                ]
                answer_texts = [f'{text}{intermediate}\n{extraction_prompt}'
                                for text, intermediate in zip(texts, intermediate_outputs)]
                input_tokens = [build_prompt(answer_text) for answer_text in answer_texts]
                inputs = tokenizer(input_tokens, padding=True, return_tensors="pt",
                                   truncation=True, max_length=2048).to('npu')
                outputs = model(**inputs, return_last_logit=True)
                logits = outputs.logits[:, -1]
                logits = logits[:, choice_tokens]
                preds = logits.argmax(dim=-1)
                correct += (preds.cpu() == batch["label"]).sum().item()
            accuracy = correct / len(dataset)
            if local_rank == 0:
                print(entry, accuracy)
            accuracy_dict[entry] = accuracy
            count_dict[entry] = len(dataset)

    acc_total, count_total = 0.0, 0
    for key in accuracy_dict:
        acc_total += accuracy_dict[key] * count_dict[key]
        count_total += count_dict[key]
    if local_rank == 0:
        print(acc_total / count_total)
        accuracy_dict.update({"acc_total": acc_total / count_total})
        json_str = json.dumps(accuracy_dict)
        with open(json_file, 'a') as f:
            f.write(json_str)


def performance(args, tokenizer, model):

    local_rank = 0 if not args.tp_size > 1 else torch.distributed.get_rank()

    if local_rank == 0:
        with open(args.performance_output_file, 'a', encoding='utf-8') as f:
            f.write(
                f"Batch,InputSeqLen(Encoding),OutputSeqLen(Decoding),TimeOfFirstToken(ms),TimePerToken(ms),TimeTotal(s),Throughput(tokens/s),ThroughputE2E(tokens/s)\n")

    if args.set_case_pair:
        seq_len_in_level = check_lists(args.seqlen_in_pair)
        seq_len_out_level = check_lists(args.seqlen_out_pair)
        assert len(seq_len_in_level) == len(seq_len_out_level)
        seq_lens = list(zip(seq_len_in_level, seq_len_out_level))
    else:
        seq_len_in_level = check_lists(args.seqlen_in_range)
        seq_len_out_level = check_lists(args.seqlen_out_range)
        seq_lens = []
        for i in range(seq_len_in_level[0], seq_len_in_level[1] + 1):
            for j in range(seq_len_out_level[0], seq_len_out_level[1] + 1):
                seq_lens.append((2 ** i, 2 ** j))

    texts = [
        "中国的首都在哪里",
        "请做一首诗歌",
        "如何学习python？",
        "五条直线相交有几个交点？",
    ]
    multi_coeff = math.ceil(args.batch / 4)
    texts = (texts * multi_coeff)[:args.batch]
    prompts = [build_prompt(text) for text in texts]
    max_seq_len_in = max(seq[0] for seq in seq_lens)

    # warm up
    if local_rank == 0:
        print("warm up start")
    inputs = tokenizer(
        prompts, return_tensors="pt", padding="max_length", max_length=max_seq_len_in).to("npu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            eos_token_id=model.config.vocab_size*2,
            max_new_tokens=4,
        )
    if local_rank == 0:
        print("warm up end")

    for seq_len_in, seq_len_out in seq_lens:

        inputs = tokenizer(
            prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len_in).to("npu")

        start_time = time.time()
        with torch.no_grad():
            torch.npu.synchronize()
            first_token_start = time.time()
            outputs = model.generate(
                **inputs,
                eos_token_id=model.config.vocab_size*2,
                max_new_tokens=1,
            )
            torch.npu.synchronize()
            first_token_end = time.time()

            torch.npu.synchronize()
            total_start = time.time()
            outputs = model.generate(
                **inputs,
                eos_token_id=model.config.vocab_size*2,
                max_new_tokens=seq_len_out,
            )
            torch.npu.synchronize()
            total_end = time.time()

        # time analysis
        time_of_first_token = (first_token_end - first_token_start)
        time_total = total_end - total_start
        time_tensor = torch.tensor([time_of_first_token, time_total], device="npu")

        if args.tp_size > 1:
            # 首token和总时间取双芯的较大值
            dist.all_reduce(time_tensor, dist.ReduceOp.MAX)

        if local_rank == 0:
            time_per_token = (time_tensor[1] - time_tensor[0]) / (seq_len_out - 1)
            throughput = args.batch * (seq_len_out - 1) / (time_tensor[1] - time_tensor[0])
            throughput_e2e = args.batch * seq_len_out / time_total

            print(
                f"batch: {args.batch}, seq_len_in: {seq_len_in}, seq_len_out: {seq_len_out}, "
                f"time_of_first_token: {time_of_first_token * 1000:.2f}ms, time_per_token: {time_per_token * 1000:.2f}ms, time_total: {time_total:.2f}s, "
                f"througput: {throughput:.2f}tokens/s, throughput_e2e: {throughput_e2e:.2f}tokens/s"
                )

            with open(args.performance_output_file, 'a', encoding='utf-8') as f:
                f.write(
                    f"{args.batch}, {seq_len_in}, {seq_len_out}, {time_of_first_token * 1000}, {time_per_token * 1000}, {time_total}, {throughput}, {throughput_e2e}\n"
                )
            
            if args.print_response:
                res = [
                    tokenizer.decode(output)
                    for output in outputs[:, inputs["input_ids"].size(1):].tolist()
                ]
                print(res)


def cli_demo(args, tokenizer, model):
    history, past_key_values = [], None
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    is_rank_0 = (args.tp_size == 1) or (torch.distributed.get_rank() == 0)

    if is_rank_0:
        print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")

    while True:
        if is_rank_0:
            objects = [input("\n用户：")]
        else:
            objects = [None]
        if args.tp_size > 1:
            torch.distributed.broadcast_object_list(objects, src=0)
        query = objects[0]
        
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            if is_rank_0:
                print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue

        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if is_rank_0:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        if is_rank_0:
            print("")


def webUI(args, tokenizer, model):
    # 跟precision类似，从github代码仓上下载，先跑通功能（符合预期），然后再尝试优化
    raise NotImplementedError


def main():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version('4.30.2')
    override_topp_and_topk()
    args = parse_args()
    tokenizer, model = get_model(args)

    if 'precision' in args.mode:
        precision(args, tokenizer, model)
    elif 'performance' in args.mode:
        performance(args, tokenizer, model)
    elif 'cli_demo' in args.mode:
        cli_demo(args, tokenizer, model)
    else:
        webUI(args, tokenizer, model)


if __name__ == '__main__':
    main()
