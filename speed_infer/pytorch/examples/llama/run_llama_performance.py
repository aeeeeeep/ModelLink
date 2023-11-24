import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch_npu
from torch_npu.contrib import transfer_to_npu
import argparse
import math

SEQ_LEN_IN = 1024
SEQ_LEN_OUT = 1024

def _init_torch_npu():
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

def set_device(device_id):
    torch.npu.set_device(torch.device(f"npu:{device_id}"))

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, padding_side='left')
    config = AutoConfig.from_pretrained(model_path)
    # padding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version not in [104, 220, 221, 222, 223]:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().npu().eval()
        model.resize_token_embeddings(len(tokenizer))
        # if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    # adapt lm_head padding weight shape
                    hs = config.hidden_size
                    lmhead_weight_offset = torch.zeros(14, hs, device=module.weight.data.device, dtype=module.weight.data.dtype)
                    # eliminate TransData op before lm_head calculation
                    module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                    module.weight.data = torch.cat((module.weight.data, lmhead_weight_offset), dim=0)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
        print("soc version: ", soc_version, " is not 910B, support NZ")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().eval()
        model.resize_token_embeddings(len(tokenizer))


    return model, tokenizer

def warm_up(model, tokenizer):
    # warm-up using huggingface's generate api
    print("--------------warm up--------------")
    test_prompt = "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", padding="max_length", max_length=128)
    with torch.no_grad():
        _ = model.generate(inputs_warm_up.input_ids.npu(), attention_mask=inputs_warm_up.attention_mask.npu(),max_new_tokens=128)

def run(model, tokenizer, file_name):
    prompts = [
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who was first president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is vice president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who was first president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is vice president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
        ]

    file_utils = open(".".join([file_name, "csv"]), 'a')
    file_utils.write(
        f"Batch,InputSeqLen(Encoding),OutputSeqLen(Decoding),TimeOfFirstToken(ms),TimePerToken(ms),TimeTotal(s),MaxSeqLen,TokensPerSecond(tps)\n")
    file_utils.close()

    for batch in [1]:

        # warm up
        print(f"batch{batch} warm up start")
        test_prompts = prompts[:batch]
        inputs = tokenizer(
            test_prompts, return_tensors="pt", padding="max_length", max_length=32)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                max_new_tokens=1024,
            )
        print(f"batch{batch} warm up end")
        print(f"batch{batch} test start:")

        for seq_len_in_level in range(5, 11):
            seq_len_in = 2 ** seq_len_in_level
            for seq_len_out_level in range(5, 11):
                seq_len_out = 2 ** seq_len_out_level

                # prepare for inputs
                test_prompts = prompts[:batch]
                inputs = tokenizer(
                    test_prompts, return_tensors="pt", padding="max_length", max_length=seq_len_in)

                with torch.no_grad():
                    # warm up encoder
                    torch.npu.synchronize()
                    outputs = model.generate(
                        inputs.input_ids.npu(),
                        attention_mask=inputs.attention_mask.npu(),
                        min_new_tokens=1,
                        max_new_tokens=1,
                    )
                    torch.npu.synchronize()
                    first_token_start = time.time()
                    outputs = model.generate(
                        inputs.input_ids.npu(),
                        attention_mask=inputs.attention_mask.npu(),
                        min_new_tokens=1,
                        max_new_tokens=1,
                    )
                    torch.npu.synchronize()
                    first_token_end = time.time()

                    torch.npu.synchronize()
                    total_start = time.time()
                    outputs = model.generate(
                        inputs.input_ids.npu(),
                        attention_mask=inputs.attention_mask.npu(),
                        min_new_tokens=seq_len_out,
                        max_new_tokens=seq_len_out,
                    )
                    torch.npu.synchronize()
                    total_end = time.time()

                # time analysis
                time_of_first_token = (first_token_end - first_token_start)
                time_total = total_end - total_start
                time_tensor = torch.tensor(
                    [time_of_first_token, time_total], device="npu")

                time_per_token = (
                    time_tensor[1] - time_tensor[0]) / (seq_len_out - 1)
                print(
                        f"batch: {batch}, seq_len_in: {seq_len_in}, seq_len_out: {seq_len_out}, time_of_first_token:{time_of_first_token * 1000}ms, time_per_token:{time_per_token * 1000}ms, time_total: {time_total}s")

                file_utils = open(".".join([file_name, "csv"]), 'a')
                file_utils.write(
                        f"{batch}, {seq_len_in}, {seq_len_out}, {time_of_first_token * 1000}, {time_per_token * 1000}, {time_total}, {seq_len_in + seq_len_out}, {1000/ time_per_token / 1000}\n"
                    )
                file_utils.close()

if __name__ == "__main__":
    _init_torch_npu()
    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--model_path",
        type=str,
        default = "/home/zhenwenqi/llama1-7b/llama1-7b",
        help="Location of Model weights, which contains model folders",)
    parser.add_argument(
        "--device_id",
        type=str,
        default=0,
        help="Choose device id",
    )
    parser.add_argument(
        "--seq_len_in",
        type=int,
        default=128,
        help="max length of input sequence",
    )
    parser.add_argument(
        "--seq_len_out",
        type=int,
        default=128,
        help="max length of input sequence",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="multi_batch_performance",
        help="file name of performance test",
    )
    args = parser.parse_args()
    set_device(args.device_id)
    print("args.model_path=",args.model_path)
    model, tokenizer = load_model(args.model_path)
    warm_up(model, tokenizer)
    run(model, tokenizer, args.file_name)

