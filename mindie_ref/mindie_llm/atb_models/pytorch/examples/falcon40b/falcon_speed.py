import os
import sys
import time
import itertools
import argparse

import tqdm
import transformers
from transformers import FalconForCausalLM, AutoTokenizer
import torch
import torch_npu

from cpu_binder import bind_cpus

RESULT_DIR = "test_result"
BATCH_SIZE_LIST = [1]
INPUT_LEN_LIST  = [32, 64, 128, 256, 512, 1024]
OUTPUT_LEN_LIST = [32, 64, 128, 256, 512, 1024]

IS_FIRST_LINE = True

if not os.path.exists(RESULT_DIR):
    try:
        os.makedirs(RESULT_DIR)
    except:
        pass

# 修改transformers的TopPLogitsWarper
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
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
    scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores

transformers.generation.TopPLogitsWarper.__call__ = __call__



def log(msg, local_rank=None):
    log_file_name = "running.log" if local_rank is None else  f"running_{local_rank}.log"
    with open(os.path.join(RESULT_DIR, log_file_name), "a") as f:
        f.write(f"{msg}\n")

def load_cache(rank=None):
    result_name = "infer_speed.csv" if rank is None else f"infer_speed_{rank}.csv"
    result_path = os.path.join(RESULT_DIR, result_name)
    if not os.path.exists(result_path):
        return set()
    with open(result_path, "r") as f:
        data = f.read().strip().split("\n")
    data = ["-".join(line.split(", ")[:3]) for line in data]
    return set(data)

def write_line_to_csv(output_file_name, batch_size, input_len, output_len, TPS='oom', RS='oom', FTT='oom', NTT='oom', rank=None):
    result_name = output_file_name if rank is None or rank == 0 else f"infer_speed_{rank}.csv"
    global IS_FIRST_LINE
    if IS_FIRST_LINE:
        col_names = ['Batch', 'MaxSeqLen', 'InputSeqLen(Encoding)', 'OutputSeqLen(Decoding)', 'ResponseTime(ms)', 'FirstTokenTime(ms)', 'TimePerTokens(ms)', 'TokensPerSecond(ms)']
        with open(result_name, "a") as f:
            f.write(f"{', '.join([str(i) for i in col_names])}\n")
        IS_FIRST_LINE = False
    
    data = [batch_size, 2048, input_len, output_len, RS, FTT, NTT, TPS]
    with open(result_name, "a") as f:
        f.write(f"{', '.join([str(i) for i in data])}\n")


def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    device_id = int(os.environ.get("SET_NPU_DEVICE", "0"))
    torch_npu.npu.set_device(local_rank + device_id)

    # numa 绑核
    bind_cpus(world_size, local_rank, device_id)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def npu_load_model_falcon_multi_card(model_weights_path, local_rank):
    ''' 当前进程只读取 local_rank 分片的权重，权重由切分脚本完成切分 '''
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

    tokenizer_path  = os.path.join(model_weights_path, "tokenizer")
    part_model_path = os.path.join(model_weights_path, "part_model", str(local_rank))

    time_start_load_model = time.time()
    log(f"[~] loading model ...", local_rank)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    model     = FalconForCausalLM.from_pretrained(part_model_path, pad_token_id=tokenizer.eos_token_id, torch_dtype=torch.float16).npu()
    log(f"[+] load model: {(time.time()-time_start_load_model)/60} min", local_rank)
    return model, tokenizer

# @torch.no_grad()
def test(model, tokenizer, output_file_name, local_rank):
    test_params    = itertools.product(BATCH_SIZE_LIST, INPUT_LEN_LIST, OUTPUT_LEN_LIST)
    total_test_num = len(BATCH_SIZE_LIST) * len(INPUT_LEN_LIST) * len(OUTPUT_LEN_LIST)
    cache_data     = load_cache(local_rank)
    process_bar    = tqdm.tqdm(total=total_test_num, initial=len(cache_data))
    device         = model.device
    case_id = 0

    prompts = [
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who was the first president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        ]

    for batch_size, input_len, output_len in test_params:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        prompt_batch = [prompts[case_id % len(prompts)] for _ in range(batch_size)]
        inputs = tokenizer(prompt_batch, return_tensors="pt", padding='max_length', max_length=input_len)
        log(f"[+] input_ids shape: {inputs.input_ids.shape}", local_rank)
        for t in inputs:
            if torch.is_tensor(inputs[t]):
                inputs[t] = inputs[t].to(device)

        with torch.no_grad():
            #  Warm up first token
            setattr(model, "total_seq_len", input_len + output_len)

            torch.npu.synchronize()
            generate_ids = model.generate(**inputs, max_new_tokens=1)
            torch.npu.synchronize()
            
            # Start inference first token
            first_token_start = time.time()
            generate_ids = model.generate(**inputs, max_new_tokens=1)
            torch.npu.synchronize()
            first_token_time = (time.time() - first_token_start) * 1000
            log(f"[+] first_token generate_ids length: {len(generate_ids[0])}", local_rank)
            
            torch.npu.synchronize()
            # Inference all tokens
            start = time.time()
            generate_ids = model.generate(**inputs, max_new_tokens=output_len)
            without_syn_time = (time.time() - start) * 1000
            torch.npu.synchronize()
            response_time = (time.time() - start) * 1000

            msg = tokenizer.batch_decode(generate_ids)
            log(f"[+] output: {msg}", local_rank)
            
            
        avg_time = (response_time - first_token_time) / (output_len - 1)
        log(f"[+] response_time: {response_time:.4f} (include syn time: {response_time-without_syn_time:.4f}), len(generate_ids) = {len(generate_ids[0])}", local_rank)
        tokens_per_second = 1000 / avg_time
        write_line_to_csv(output_file_name, batch_size, input_len, output_len, TPS=tokens_per_second, RS=response_time, FTT=first_token_time, NTT=avg_time, rank=local_rank)
        process_bar.update(1)
        case_id += 1
    process_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--model_path",
        type=str,
        default = "/home/weights/falcon40b_4cards",
        help="Location of Model weights, which contains model folders",)

    parser.add_argument(
        "--file_name",
        type=str,
        default = "1_batch_performance_falcon40b.csv",
        help="",)
    args = parser.parse_args()
    output_file_name = args.file_name
    model_path = args.model_path
    log(f"[~] setup_model_parallel")

    local_rank, world_size = setup_model_parallel()
    model, tokenizer = npu_load_model_falcon_multi_card(model_path, local_rank)
    device = model.device
    log(f"[+] model device: {device}", local_rank)
    test(model, tokenizer, output_file_name, local_rank)
