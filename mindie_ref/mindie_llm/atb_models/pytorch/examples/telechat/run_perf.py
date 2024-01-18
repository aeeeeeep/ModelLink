import os
import time
import argparse

import torch
import torch_npu
from transformers import AutoTokenizer, TelechatForCausalLM, TelechatConfig

torch.npu.set_device(torch.device(f"npu:0"))
soc_version = torch_npu._C._npu_get_soc_version()

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)


def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group('EVAL Task Parameters')
    group.add_argument(
        '--model_path', type=str)
    group.add_argument(
        '--quant_path', type=str)
    args_input = parser.parse_args()
    return args_input


args = get_args()

os.setenv('QUANT_PATH', args.quant_path)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
config = TelechatConfig.from_pretrained(args.model_path)
model = TelechatForCausalLM.from_pretrained(args.model_path, config=config).eval().half().npu()

soc_version = torch_npu._C._npu_get_soc_version()
if soc_version in [105, 220, 221, 222, 223]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
else:
# if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == 'lm_head':
                # eliminate TransData op before lm_head calculation
                module.weight = torch.nn.parameter.Parameter(module.weight.data)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data.transpose(0, 1).contiguous(), 29)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)


def performance_test():
    context = "<_user>好的<_bot>"
    batch_sizes = [1]
    input_len = [10, 100, 500, 1000, 2000]
    output_len = [10, 100, 500, 1000, 2000]
    test_cases = [(bs, inseq, outsq) for bs in batch_sizes for inseq in input_len for outsq in output_len]

    print("=======================warm up=====================")
    context_ids = tokenizer(context, return_tensors="pt", max_length=input_len[0],
                                    padding='max_length', truncation=True)
    output = model.generate(context_ids["input_ids"].to(0), max_new_tokens=output_len[0], do_sample=False, use_cache=True,
                                    repetition_penalty=1.03)
    
    print("=======================inference=====================")
    for batch_size, input_seq_len, output_seq_len in test_cases:
        print(f"batch_size is {batch_size}, input_seq_len is {input_seq_len}, output_seq_len is {output_seq_len}")
        context_ids = tokenizer(context, return_tensors="pt", max_length=input_seq_len,
                                    padding='max_length', truncation=True)

        torch.npu.empty_cache()
        with torch.no_grad():
            start_time_model = time.time()
            output = model.generate(context_ids["input_ids"].to(0), max_new_tokens=output_seq_len, do_sample=False, use_cache=True,
                                    repetition_penalty=1.03)
            print(f"lenoutput{len(output[0])}, outputseqlen{output_seq_len}")
            end_time_model = time.time()
            output_str = tokenizer.decode(output[0].tolist()).split("<_bot>")[-1]
            end_time_e2e = time.time()
            print("model_time:", end_time_model - start_time_model, "s")
            print("end2end_time:", end_time_e2e - start_time_model)
            print("output token delay", output_seq_len / (end_time_model - start_time_model), "token/s")

performance_test()