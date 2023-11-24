import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch_npu
from torch_npu.contrib import transfer_to_npu
import argparse

SEQ_LEN_IN = 128
SEQ_LEN_OUT = 128


def _init_torch_npu():
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

def set_device(device_id):
    torch.npu.set_device(torch.device(f"npu:{device_id}"))

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, padding_side='left')
    # 此处model不要.npu(),modeling里有合并权重的操作，会占用很大内存
    
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
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", padding="max_length", max_length=1024)
    with torch.no_grad():
        _ = model.generate(inputs_warm_up.input_ids.npu(), attention_mask=inputs_warm_up.attention_mask.npu(), max_new_tokens=1024)

def run(model, tokenizer, seq_len_in, seq_len_out):
    print("--------------inference--------------")
    prompt = ["Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"]
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=1024)
    start_time = time.time()
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids.npu(), 
                                    attention_mask=inputs.attention_mask.npu(), 
                                    max_new_tokens=seq_len_out)
    end_time = time.time()
    # decode
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for item in res:
        print(item)
    # time analysis
    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    elapse = end_time - start_time
    print(f"Output generated in {elapse:.2f}s, {(new_tokens/elapse):.2f} tokens/s, {new_tokens} new tokens generated.")

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
    args = parser.parse_args()
    set_device(args.device_id)
    model, tokenizer = load_model(args.model_path)
    warm_up(model, tokenizer)
    run(model, tokenizer, args.seq_len_in, args.seq_len_out)

