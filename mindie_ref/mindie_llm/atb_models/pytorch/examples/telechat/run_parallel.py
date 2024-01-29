import os, time
from tqdm import tqdm
import torch
import pandas as pd
import argparse
import jsonlines
import torch_npu
#from torch_npu.contrib import transfer_to_npu
from accelerate import init_empty_weights, load_checkpoint_in_model, dispatch_model, infer_auto_device_map, \
    load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TelechatForCausalLM, TelechatConfig


def load_model(args):
    torch.npu.set_device(torch.device(f"npu:0"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = TelechatConfig.from_pretrained(args.checkpoint)

    with init_empty_weights():
        model = TelechatForCausalLM._from_config(config)
    model.tie_weights()

    max_mem = 4686198491 * 8  # 32G
    device_map = infer_auto_device_map(
        model.transformer,
        max_memory={0: max_mem},
        no_split_module_classes=["BloomBlock"],
        dtype='float16'
    )

    load_checkpoint_in_model(
        model.transformer,
        args.checkpoint,
        device_map=device_map,
        offload_folder=None,
        dtype='float16',
        # offload_state_dict=True
    )
    model.tie_weights()

    full_model_device_map = {f"transformer.{k}": v for k, v in device_map.items()}
    full_model_device_map["lm_head"] = 0
    dispatch_model(model, device_map=full_model_device_map)
    print("ok!!")

    model.eval().npu()
    return model, tokenizer

def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if local_rank==0:
        torch_npu.npu.set_device(0)
    elif local_rank==1:
        torch_npu.npu.set_device(1)
    torch.manual_seed(1)
    return local_rank, world_size

def load_model_parallel(args):
    local_rank, world_size = setup_model_parallel()

    tokenizer_path = os.path.join(args.checkpoint, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    part_model_path=os.path.join(args.checkpoint, "part_model", str(local_rank))
    config = TelechatConfig.from_pretrained(part_model_path)
    model = TelechatForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16).npu()
    if not local_rank:
        print("load model successfully!")
    
    # with init_empty_weights():
    #     model = TelechatForCausalLM._from_config(config)
    # model.tie_weights()

    # max_mem = 4686198491 * 8  # 32G

    # device_map = infer_auto_device_map(
    #     model.transformer,
    #     max_memory={0: max_mem},
    #     no_split_module_classes=["BloomBlock"],
    #     dtype='float16'
    # )
    # print(type(model))
    # load_checkpoint_in_model(
    #     model.transformer,
    #     part_model_path,
    #     device_map=device_map,
    #     offload_folder=None,
    #     dtype='float16',
    #     # offload_state_dict=True
    # )
    # model.tie_weights()

    # full_model_device_map = {f"transformer.{k}": v for k, v in device_map.items()}
    # full_model_device_map["lm_head"] = 0
    # dispatch_model(model, device_map=full_model_device_map)
    # print("ok!!")

    model.eval().npu()
    return model, tokenizer

def nd2nz(model):
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [105, 220, 221, 222, 223]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data,2)
    else:
        # if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types 
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    # eliminate TransData op before lm_head calculation
                    module.weight = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data.transpose(0,1).contiguous(), 29)
                # module.weight.data = torch_npu.npu_format_cast(module.weight.data,29)

    for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data,2)

def load_dataset(input_path):
    f = jsonlines.open(input_path, "r")
    questions = []
    origin = []
    for data in f:
        questions.append(data["input"])
    f.close()
    return questions

def print_(is_printing, msg):
    if is_printing:
        print(msg)


def infer_precision():
    args = get_args()
    print("start!")
    questions = load_dataset(args.input_path)
    is_printing = 1
    if args.run :

        model, tokenizer = load_model(args)
    if args.runparallel:
        model, tokenizer = load_model_parallel(args)
        is_printing = torch.distributed.get_rank()


    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)
    nd2nz(model)

    answers = []
    for input_str in tqdm(questions):
        context = "<_user>" + str(input_str) + "<_bot>"
        context_ids = tokenizer(context, return_tensors="pt")

        with torch.no_grad():
            start_time_model = time.time()
            output = model.generate(context_ids["input_ids"].npu(), max_length=4096, do_sample=False, use_cache=True,
                                    repetition_penalty=1.03, eos_token_id=[160133, 160130])
            end_time_model = time.time()
            output_str = tokenizer.decode(output[0].tolist()).split("<_bot>")[-1]
            print_(is_printing, output_str)
            end_time_e2e = time.time()
            answers.append(output_str)
            model.transformer.clear_cache()
        # print("model_time:", end_time_model - start_time_model, "s")
        # print("end2end_time:", end_time_e2e - start_time_model)
        # print("output token delay", len(output_str[0]) / (end_time_model - start_time_model), "s")

    f = jsonlines.open(args.output_path, "w")
    for q, i in zip(questions, answers):
        f.write({"input": q, "infer": i})
    f.close()


def performance_test():
    args = get_args()
    print("start!")
    questions = load_dataset(args.input_path)
    is_printing = 1
    if args.run :
        model, tokenizer = load_model(args)
    if args.runparallel:
        model, tokenizer = load_model_parallel(args)
        is_printing = torch.distributed.get_rank()

    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)
    nd2nz(model)

    context = "<_user>好的<_bot>"
    batch_size = [1, 4, 8, 16]
    input_len = [256, 512, 1024]

    test_cases = [(bs, inseq, inseq) for bs in batch_size for inseq in input_len]

    print("============ warm up =================")
    context_ids = tokenizer(context, return_tensors="pt", max_length=input_len[0], padding='max_length', truncation=True)
    output = model.generate(context_ids["input_ids"].to("npu"), max_new_tokens=input_len[0], do_sample=False, use_cache=True,
                                    repetition_penalty=1.03, eos_token_id=[160133, 160130])
    csv_name = f"performance_telechat_device{torch_npu.npu.current_device()}.csv"
    file_utils = open(csv_name, 'a')
    file_utils.write(f"Batch,Input_seq_len,Output_seq_len,TotalTime(s),avg_time_pre_token(ms),token_per_second(tps)\n")
    file_utils.close()
    print("================== inference ===================")
    for bs, in_seq, out_seq in test_cases:
        print("bs, in_seq, out_seq")
        print(bs, in_seq, out_seq)
        context_lst = [context for _ in range(bs)]
        context_ids = tokenizer(context_lst, return_tensors="pt", max_length=out_seq, padding='max_length', truncation=True)

        torch.npu.empty_cache()
        with torch.no_grad():
            torch.npu.synchronize()
            start_time_model = time.time()
            output = model.generate(context_ids["input_ids"].npu(),  max_new_tokens=out_seq, min_new_tokens=out_seq, do_sample=False, use_cache=True,
                                    repetition_penalty=1.03, eos_token_id=[160133, 160130])
            torch.npu.synchronize()
            end_time_model = time.time()
            output_str = tokenizer.batch_decode(output)
            torch.npu.synchronize()
            end_time_e2e = time.time()

            time_generate = end_time_model - start_time_model
            time_per_token = time_generate * 1000 / out_seq
            token_per_seconds = bs * out_seq / time_generate
            print_(is_printing, f"len: {out_seq}")
            print_(is_printing, output_str)
            print_(is_printing, f"model time : {time_generate}s")
            print_(is_printing, f"end2end time: {end_time_e2e - start_time_model}s")
            print_(is_printing, f"time per token: {time_per_token}ms")
            print_(is_printing, f"token_per_seconds: {token_per_seconds}tps")

            file_utils = open(csv_name, 'a')
            file_utils.write(f"{bs},{in_seq},{out_seq},{time_generate:.2f},{time_per_token:.2f},{token_per_seconds:.2f}\n")
            file_utils.close()




def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group('EVAL Task Parameters')
    group.add_argument(
        '--device', type=str, metavar='DIR', default=None)
    group.add_argument(
        '--input-path', type=str, metavar='DIR', default="/home/HwHiAiUser/workspace/data/test.jsonl")        
    group.add_argument(
        '--output-path', type=str, metavar='DIR', default="/home/HwHiAiUser/workspace/data/test-out.jsonl")
    group.add_argument(
        '--checkpoint', type=str, metavar='DIR', default="/home/telechat_anti/telechat_anti_cpu_part/")
    group.add_argument(
        '--tokenizer-path', type=str, metavar='DIR', default="/home/telechat_anti/telechat_anti_cpu_part/tokenizer/")
    group.add_argument(
        '--run', action = "store_true", help = "run float model")
    group.add_argument(
        '--runparallel', action = "store_true", help = "run parallel+float model")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    infer_precision()
    performance_test()