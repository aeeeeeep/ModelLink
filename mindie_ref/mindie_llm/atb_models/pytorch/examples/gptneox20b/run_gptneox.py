import argparse
import os
import random
import time

import numpy as np
import torch
import torch_npu
from transformers import AutoTokenizer

from configuration_gpt_neox import GPTNeoXConfig

DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
device_id = 0
if DEVICE_ID is not None:
    device_id = int(DEVICE_ID)
print(f"[WARNING] USE npu:{device_id}")
torch.npu.set_device(torch.device(f"npu:{device_id}"))
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril,SoftmaxV2,LayerNormGrad,ReduceProd"
torch.npu.set_option(option)
seed = 128


def get_args():
    parser = argparse.ArgumentParser(
        description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default="./",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--precision",
        default=False,
        type=bool,
        help="Location of Model weights, which contains model folders",
    )
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_npu.npu.is_available():
        torch_npu.npu.manual_seed_all(seed)


set_random_seed(seed)


def main_full_model(is_prof: bool = False):
    from patches.models.modeling_gpt_neox_model_flashattention_performance_v2 import GPTNeoXForCausalLM

    print("Done load tokenizer", time.time())

    config = GPTNeoXConfig.from_pretrained(model_path)
    config.is_decoder = True
    print("Done load model config in cpu", config)
    model = GPTNeoXForCausalLM.from_pretrained(model_path, config=config)
    print("Done load model in cpu", time.time())

    # clear npu cache
    torch_npu.npu.empty_cache()
    torch_npu.npu.reset_peak_memory_stats()

    model.gradient_checkpointing_disable()
    model.eval()
    model.half().npu()

    # now_memory = torch_npu.npu.memory_stats()
    # print("After init model on npu memory stats is", now_memory)
    torch_npu.npu.synchronize()
    peak_memory = torch_npu.npu.max_memory_allocated()
    print("Done load model to device", time.time(), "peak_memory", peak_memory)

    input_token = ["a"] * 128
    prompt = " ".join(input_token)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.npu()

    print("Start warm up....")
    start = time.time()
    output = model(inputs)

    torch_npu.npu.synchronize()
    peak_memory = torch_npu.npu.max_memory_allocated()
    print("Done warm up", time.time() - start, "peak_memory", peak_memory)

    torch_npu.npu.empty_cache()
    torch_npu.npu.reset_peak_memory_stats()

    start = time.time()
    output_ids = model.generate(inputs, do_sample=False, max_new_tokens=128)
    torch_npu.npu.synchronize()
    peak_memory = torch_npu.npu.max_memory_allocated()
    print("generate", time.time() - start, "peak_memory", peak_memory)

    output_str = tokenizer.batch_decode(output_ids)[0]


def main_small_model():
    from patches.models.modeling_gpt_neox_model_flashattention_performance_v2 import GPTNeoXForCausalLM

    config = GPTNeoXConfig(num_hidden_layers=2, is_decoder=True)
    print("==config", config)

    model = GPTNeoXForCausalLM(config)
    print("==Done init model", time.time())
    model.gradient_checkpointing_disable()
    model.eval()
    model.half().npu()
    print("==Done to device", time.time())

    inputs = tokenizer("My favorite food is fired fished, but it hash lots of fat.",
                       return_tensors="pt").input_ids.npu()

    output_ids = model.generate(inputs, do_sample=False, max_new_tokens=5)


def main_chat():
    if args.precision:
        from patches.models.modeling_gpt_neox_model_precision import GPTNeoXForCausalLM
    else:
        from patches.models.modeling_gpt_neox_model import GPTNeoXForCausalLM
    start_time = time.time()
    print("Start infer: ", start_time)

    config = GPTNeoXConfig.from_pretrained(model_path)
    config.is_decoder = True
    print(f"Done load model config in cpu cost: {time.time() - start_time}s, ", config)
    model = GPTNeoXForCausalLM.from_pretrained(model_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Done load model in cpu, cost: {time.time() - start_time}s")

    model.gradient_checkpointing_disable()
    model.eval()
    model.half().npu()

    with torch.no_grad():
        prompts = [
            "hello, please introduce yourself.",
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:"
        ]
        inputs = tokenizer(prompts, padding="max_length", max_length=1024, return_tensors="pt")
        print("inputs is", inputs)
        print("-------warm up------")
        _ = model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(), do_sample=False,
                           max_new_tokens=10)

        print("-------inference------")
        output_ids = model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(), do_sample=False,
                                    max_new_tokens=30)

        print("output ids is", output_ids)
        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(f"Done infer: cost: {time.time() - start_time}s")
        for answer in answers:
            print("answer is: ", answer, end="\n\n")


if __name__ == '__main__':
    args = get_args()
    model_path = args.load_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    main_chat()
