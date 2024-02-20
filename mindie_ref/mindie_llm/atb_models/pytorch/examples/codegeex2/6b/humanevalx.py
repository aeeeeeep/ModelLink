# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2031. All rights reserved

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers.utils import check_min_version
from torch_npu.contrib import transfer_to_npu

sys.path.insert(0, Path(__file__, '../../../chatglm2/6b').resolve())
from main import get_model
sys.path.insert(0, Path(__file__,  '../CodeGeeX2/evaluation').resolve())
from generation import CodeStoppingCriteria
from utils import Logger, read_dataset, process_extra_prompt, is_code_generation_finished, cleanup_code


def parse_args():
    parser = argparse.ArgumentParser(description="Adapting CodeGeeX2-6B on Ascend")
    parser.add_argument("--model_path", type=str, required=True, help="The path to model weights")
    parser.add_argument("--tp_size", type=int, default=1, help="Whether test model in parallel")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument(
        "--model_file",
        type=str,
        default="../../chatglm2/6b/patches/models/modeling_chatglm_ascend.py",
        help="The implementation of model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='',
        help="The path to dataset"
    )
    parser.add_argument(
        '--output_dir',
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Max sequence length",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p Probability for sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k for sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--greedy",
        type=int,
        default=0,
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Micro batch size for each GPU",
    )
    parser.add_argument(
        "--samples_per_problem",
        type=int,
        default=200,
        help="Number of samples to generate for each problem",
    )
    parser.add_argument(
        '--language_type',
        default="python",
        help='Identify the type of programming language to generate',
    )
    parser.add_argument(
        '--generation_mode',
        default="instruction",
    )

    return parser.parse_args()


def precision(args, tokenizer, model):
    local_rank = 0 if not args.tp_size > 1 else torch.distributed.get_rank()

    dataset_type = 'humanevalx'
    entries = read_dataset(args.dataset, dataset_type=dataset_type)
    for entry in entries.values():
        entry["prompt"] = process_extra_prompt(
            entry["prompt"], 
            language_type=args.language_type, 
            dataset_type=dataset_type, 
            generation_mode=args.generation_mode,
        )
    res = []
    for entry in entries.values():
        res.extend([entry] * (args.samples_per_problem // args.micro_batch_size))
    output_dir = args.output_dir
    os.makedirs(output_dir, mode=0o640, exist_ok=True)
    with os.fdopen(
        os.open(f"{output_dir}/results.jsonl", os.O_WRONLY | os.O_CREAT, 0o640), 
        "w", encoding="utf-8") as fout:
        with torch.no_grad():
            for entry in tqdm(res):
                prompt = entry["prompt"]
                inputs = tokenizer([prompt for _ in range(args.micro_batch_size)], return_tensors="pt").to(model.device)
                stop_criteria = CodeStoppingCriteria(
                    max_length=args.max_length,
                    micro_batch_size=args.micro_batch_size,
                    tokenizer=tokenizer,
                    dataset_type=dataset_type,
                    language_type=args.language_type,
                    prompt=prompt)
                outputs = model.generate(**inputs,
                                        max_length=args.max_length,
                                        do_sample=True if not args.greedy else False,
                                        use_cache=True,
                                        stopping_criteria=[stop_criteria],
                                        top_p=args.top_p,
                                        top_k=args.top_k,
                                        temperature=args.temperature,
                                        pad_token_id=tokenizer.eos_token_id)
                if local_rank == 0:
                    for output in outputs:
                        response = tokenizer.decode(output)
                        entry["generation_raw"] = response
                        entry["generation"] = cleanup_code(
                            response[len(prompt):], 
                            dataset_type=dataset_type,
                            language_type=args.language_type)
                        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        fout.flush()


def main():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version('4.30.2')
    args = parse_args()

    tokenizer, model = get_model(args)
    precision(args, tokenizer, model)


if __name__ == '__main__':
    main()
