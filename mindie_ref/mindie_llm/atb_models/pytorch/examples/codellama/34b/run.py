import os
import time
from tqdm import tqdm
import torch
import pandas as pd
import argparse
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

def load_model(args):
    torch.npu.set_device(torch.device(f"npu:{args.device}"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = (
        AutoModelForCausalLM.from_pretrained(args.checkpoint)
        .eval()
        .half()
        .npu()
    )
    return model, tokenizer


def setup_model_parallel(args):
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_device = int(args.device) + int(local_rank)
    print(f"loading model on device {local_device}, rank: {local_rank}")
    torch_npu.npu.set_device(local_device)
    torch.manual_seed(1)
    return local_rank, world_size


def load_model_parallel(args):
    local_rank, world_size = setup_model_parallel(args)

    tokenizer_path = os.path.join(args.checkpoint, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    part_model_path = os.path.join(args.checkpoint, "part_model", str(local_rank))
    model = (
        AutoModelForCausalLM.from_pretrained(part_model_path)
        .eval()
        .half()
        .npu()
    )
    if not local_rank:
        print("load model successfully!")
    return model, tokenizer

def get_prompt(message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

def nd2nz(model):
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [105, 220, 221, 222, 223]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
    else:
        # if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == "lm_head":
                    # eliminate TransData op before lm_head calculation
                    module.weight = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = torch_npu.npu_format_cast(
                    module.weight.data.contiguous(), 29
                )

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)



def print_(is_printing, msg, end="\n"):
    if is_printing:
        print(msg, end=end)


def infer_precision(query, model, tokenizer, is_printing, prompt=False):
    system = DEFAULT_SYSTEM_PROMPT
    if prompt:
        prompts = get_prompt(query, system)
    else:
        prompts = query
    gen_kwargs = dict(
        max_new_tokens=512,
        bos_token_id=1,
        top_p=0.9,
        top_k=50,
        temperature=0.1,
        do_sample=False,
        num_beams=1,
    )
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False)
    input_id = inputs["input_ids"]
    generation_kwargs = dict(
        inputs=inputs["input_ids"].npu(),
        attention_mask=inputs['attention_mask'].npu(),
        eos_token_id=tokenizer.eos_token_id,
        # repetition_penalty=1.1,
        **gen_kwargs
    )
    print(f"问:{query}")

    with torch.no_grad():
        torch.npu.synchronize()
        start_time_model = time.time()
        output = model.generate(
            **generation_kwargs
        )
        output_str = tokenizer.decode(output[0].tolist())
        torch.npu.synchronize()
        end_time_model = time.time()
        print(f'CodeLlama34b-答:\n')
        print_(is_printing, output_str)
    print(f"model_time:{end_time_model - start_time_model}s")
    print(
        f"token avg delay {(end_time_model - start_time_model) * 1000 / len(output[0].tolist()):.2f}ms"
    )


def get_args():
    parser = argparse.ArgumentParser(
        "Evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_argument_group("EVAL Task Parameters")
    group.add_argument("--device", type=str, metavar="DIR", default=None)
    group.add_argument("--input-path", type=str, metavar="DIR", default="")
    group.add_argument("--output-path", type=str, metavar="DIR", default="")
    group.add_argument("--checkpoint", type=str, metavar="DIR", default="")
    group.add_argument("--run-single", action="store_true", help="run float model")
    group.add_argument(
        "--run-parallel", action="store_true", help="run parallel + float model"
    )
    group.add_argument("--run-precision", action="store_true", help="run precision")
    group.add_argument("--run-performance", action="store_true", help="run performance")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args_main = get_args()

    args = get_args()
    print("start!")
    is_printing = 1
    if args.run_single:
        model, tokenizer = load_model(args)
    if args.run_parallel:
        model, tokenizer = load_model_parallel(args)
        is_printing = torch.distributed.get_rank() == 0

    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)
    # nd2nz(model)

    if args_main.run_precision:
        infer_precision("Please use Python to implement a binary search algorithm.", model, tokenizer, is_printing, prompt=False)
        infer_precision("Please use Python to implement a binary search algorithm.", model, tokenizer, is_printing, prompt=True)
    elif args_main.run_performance:
        performance_test()
