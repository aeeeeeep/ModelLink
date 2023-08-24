"""Sample Generate LLAMA"""
import os
import time

import torch
from torch import distributed as dist
from transformers import LlamaTokenizer
from deepspeed.accelerator import get_accelerator

from ascendspeed import get_args
from ascendspeed.model import LlamaForCausalLM
from ascendspeed.initialize import initialize_megatron


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--temperature", type=float, default=0.5, help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False, help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.9, help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
    group.add_argument("--max_new_tokens", type=int, default=128, help='Size of the output generated text.')
    return parser


def task1():
    prompt = "请用python实现一个快速排序算法"
    instruction = template.format(instruction=prompt)

    t = time.time()
    output = model.generate(
        instruction,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False
    )

    if dist.get_rank() == 0:
        print("===========================================")
        print(f"\nYou:\n{prompt}\n\nMegatron-LM:\n{output}")
        print("===========================================")
        print(f"\nElapsed: {round(time.time() - t, 2)}s")

    dist.barrier()


def task2():
    while True:
        instruction = ""
        terminate_runs = torch.zeros(1, dtype=torch.int64, device=torch.device(get_accelerator().device_name()))
        if dist.get_rank() == 0:
            prompt = input("You >> ")
            if prompt.strip() in ["q", "exit", "quit"]:
                terminate_runs += 1

            instruction = template.format(instruction=prompt)

        dist.all_reduce(terminate_runs)
        if terminate_runs > 0:
            break

        responses = model.generate(
            instruction,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        for output in responses:
            if dist.get_rank() == 0:
                os.system("clear")
                print(f"You:\n{prompt}\nMegatron-LM:\n{output}")


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    model = LlamaForCausalLM.from_pretrained(args.load)
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path)

    template = "Below is an instruction that describes a task. Write a response that appropriately " \
               "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    # task1()
    task2()
