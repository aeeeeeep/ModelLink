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
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().npu().eval()
    config = AutoConfig.from_pretrained(model_path)
    # padding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version not in [104, 220, 221, 222, 223]:
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

    return model, tokenizer

def warm_up(model, tokenizer):
    # warm-up using huggingface's generate api
    print("--------------warm up--------------")
    test_prompt = "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", padding="max_length", max_length=SEQ_LEN_IN)
    with torch.no_grad():
        _ = model.generate(inputs_warm_up.input_ids.npu(), attention_mask=inputs_warm_up.attention_mask.npu(), max_new_tokens=SEQ_LEN_OUT)

def precision(model, tokenizer, seq_len_in, seq_len_out,batch):
    print("--------------inference--------------")
    prompts = [
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
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
    print("batch=",batch)
    for i in range(2):
        print("current_question =", i)
        inputs = tokenizer(prompts[i:i+batch], return_tensors="pt", padding="max_length", max_length=seq_len_in)
        with torch.no_grad():
            generate_ids = model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(), max_new_tokens=seq_len_out)
        res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for item in res:
            print(item)
if __name__ == "__main__":

    _init_torch_npu()

    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--model_path",
        type=str,
        default = "./",
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
        "--batch",
        type=int,
        default="2",
        help="Set batch_size, the maximum value is 32",
    )
    args = parser.parse_args()
    set_device(args.device_id)
    model, tokenizer = load_model(args.model_path)
    warm_up(model, tokenizer)
    precision(model, tokenizer, args.seq_len_in, args.seq_len_out,args.batch)

