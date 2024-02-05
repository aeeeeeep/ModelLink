# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import time
from functools import partial
from itertools import product
from tempfile import NamedTemporaryFile

import pandas as pd
import psutil
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_falcon_ascend import FalconForCausalLM



def get_rank():
    return 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()


def is_rank_0():
    return get_rank() == 0


def print_rank_0(*p_args, **kwargs):
    if is_rank_0():
        print(*p_args, **kwargs)


def get_world_size():
    return 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()


def add_test_args(parser):
    group = parser.add_argument_group(title='test case info')
    group.add_argument("--batch", type=int, default=1, help="batch size")
    group.add_argument(
        "--set_case_pair",
        action='store_true',
        help="weather to use case_pair",
    )
    group.add_argument(
        "--seqlen_in_range",
        default=[5, 10],
        type=int,
        nargs='+',
        help="input seqlen ranges from 2^5 to 2^10",
    )
    group.add_argument(
        "--seqlen_out_range",
        default=[5, 10],
        type=int,
        nargs='+',
        help="output seqlen ranges from 2^5 to 2^10",
    )
    group.add_argument(
        "--seqlen_in_pair",
        type=int,
        nargs='+',
        default=[256, 512, 1024],
        help="specified case",
    )
    group.add_argument(
        "--seqlen_out_pair",
        type=int,
        nargs='+',
        default=[64, 128, 256],
        help="specified case",
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Bloom7B arguments.")
    
    add_test_args(parser)

    parser.add_argument(
        "--model_path",
        default="./",
        help="Location of Model weights, which contains model folders",
    )

    parser.add_argument(
        "--data_dtype",
        default="int8",
        choices=['fp16', 'int8'],
        help="Data dtype",
    )

    parser.add_argument(
        "--device",
        nargs='+',
        type=int,
        default=[1],
        help="NPU devices",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default='310',
        choices=['310', '910'],
        help="Ascend hardwares"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='performance',
        choices=['performance', 'precision'],
        help="Specify the mode in which to run the script"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='./',
        help="The path to eval dataset"
    )
    parser.add_argument(
        "--performance_output_file",
        type=str,
        default='performance.csv',
        help="file name of performance test"
    )

    p_args = parser.parse_args()
    return p_args


def load_model(load_args):
    torch_npu.npu.set_device(load_args.device[0])
    tokenizer = AutoTokenizer.from_pretrained(load_args.model_path, use_fast=False)
    model = FalconForCausalLM.from_pretrained(load_args.model_path, trust_remote_code=True).half().npu().eval()
    model.generate = partial(model.generate, pad_token_id=tokenizer.eos_token_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # 使用二进制优化，消除动态shape的编译问题
    torch.npu.set_compile_mode(jit_compile=False)

    # model = cast_nz_weight(model)
    model.eval()
        
    return model, tokenizer


def inference(infer_model, infer_tokenizer, prompt, batch, seqlen_in, seqlen_out):
    prompt = ["Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:"] * batch
    # warmup
    inputs = infer_tokenizer(prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen_in)
    with torch.no_grad():
        torch.npu.synchronize()
        _ = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=2 if seqlen_out > 1 else 1)
        torch.npu.synchronize()
    
    print_rank_0("inference start")
    # tokenize
    torch.npu.synchronize()
    start_time = time.time()
    inputs = infer_tokenizer(prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen_in)
    # infer
    with torch.no_grad():
        torch.npu.synchronize()
        first_token_start = time.time()
        _ = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=1)
        torch.npu.synchronize()
        first_token_end = time.time()
        torch.npu.synchronize()
        generate_start = time.time()
        generate_ids = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=seqlen_out)
        torch.npu.synchronize()
        generate_end = time.time()

    # decode
    res = infer_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    torch.npu.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    
    # time analysis
    time_of_first_token = (first_token_end - first_token_start) * 1000
    time_generate = (generate_end - generate_start) * 1000
    time_tensor = torch.tensor(
    [time_of_first_token, time_generate, total_time]).npu()

    if get_world_size() >= 2:
        torch.distributed.all_reduce(time_tensor, torch.distributed.ReduceOp.MAX)

    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    time_per_token = ((time_tensor[1] - time_tensor[0]) / (new_tokens - 1)) if new_tokens - 1 else 0
    time_of_first_token = time_tensor[0]
    total_time = time_tensor[2]

    print_rank_0("\nQ&A results are as follows:")
    for idx, item in enumerate(res):
        print_rank_0(f"\n[Q&A {idx+1}]\n", item)
    # time analysis
    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    print_rank_0(f"\nBatch: {batch}, Input tokens number: {len(inputs.input_ids[0])}, Output tokens number: {new_tokens}")
    print_rank_0(f"Generated in {total_time:.2f} s, first token costs {time_of_first_token:.2f} ms, avg token cost {time_per_token:.2f} ms, inference speed is {(batch*new_tokens/total_time):.2f} tokens/s")

    return time_of_first_token.cpu().tolist(), time_per_token.cpu().tolist(), total_time.cpu().tolist()



def performance_test(perf_args):
    args = perf_args
    result_filename = args.performance_output_file
    model, tokenizer = load_model(args)

    prompt = ""

    if args.set_case_pair:
        assert len(args.seqlen_in_pair) == len(args.seqlen_out_pair), "length between seqlen_in_pair and seqlen_out_pair not equal!"
        test_cases = [(args.batch, seq_len_in, seq_len_out) for seq_len_in, seq_len_out in zip(args.seqlen_in_pair, args.seqlen_out_pair)]
    else:
        assert len(args.seqlen_in_range) == len(args.seqlen_out_range) == 2, "length for seqlen_in_range and seqlen_out_range which is [begin, end]!"
        batch_sizes = [args.batch]
        seq_lens_in = [2**x for x in range(args.seqlen_in_range[0], args.seqlen_in_range[1] + 1)]
        seq_lens_out = [2**x for x in range(args.seqlen_out_range[0], args.seqlen_out_range[1] + 1)]
        test_cases = product(*[batch_sizes, seq_lens_in, seq_lens_out])
    
    results = {"Batch": [], "MaxSeqLen": [], "InputSeqLen(Encoding)": [], "OutputSeqLen(Decoding)": [], "ResponseTime(s)": [], "FirstTokenTime(ms)": [], "TimePerTokens(ms)": []}
    for batch_size, seq_len, max_new_tokens in test_cases:
        setattr(model, "total_seq_len", seq_len + max_new_tokens)
        first_token, per_token, total_time = inference(model, tokenizer, prompt, batch_size, seq_len, max_new_tokens)
        results["Batch"].append(batch_size)
        results["MaxSeqLen"].append(seq_len + max_new_tokens)
        results["InputSeqLen(Encoding)"].append(seq_len)
        results["OutputSeqLen(Decoding)"].append(max_new_tokens)
        results["ResponseTime(s)"].append(total_time)
        results["FirstTokenTime(ms)"].append(first_token)
        results["TimePerTokens(ms)"].append(per_token)

        # save every case
        df = pd.DataFrame(results)
        if is_rank_0():
            df.to_csv(result_filename)
    print_rank_0(f"save performance to {result_filename}")


def precision_test(prec_args):
    try:
        from atb_speed.common.precision import get_precision_test_cls
        from atb_speed.common.config import atb_speed_config
    except ImportError as e:
        print_rank_0("you need to install atb_speed sdk!")
        raise Exception("you need to install atb_speed sdk!")


    class Falcon7B:
        def __init__(self, init_args):
            self.model, self.tokenizer = load_model(init_args)
            self.local_rank = get_rank()

    with NamedTemporaryFile('w+t') as f:
        # Read/write to the file
        f.write('[precision]\n')
        f.write('mode=mmlu\n')
        f.flush()

        atb_speed_config.init_config(f.name)
    falcon7b = Falcon7B(prec_args)
    c_t = get_precision_test_cls()(falcon7b, prec_args.dataset_path, batch=prec_args.batch)
    c_t.run()


def main(args):
    if args.mode == "performance":
        performance_test(args)
    elif args.mode == "precision":
        precision_test(args)
    else:
        raise Exception("mode error!")


if __name__ == "__main__":
    main_args = get_args()
    main(main_args)