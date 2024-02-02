# coding=utf-8
import argparse
import os
import time
import shutil
from itertools import product
import numpy as np
import torch
import torch_npu
import transformers
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, TelechatForCausalLM, TelechatConfig, AutoConfig

def get_args():
    parser = argparse.ArgumentParser(description="Telechat info.")
    parser.add_argument("--world-size", type=int, default=2, help="world size")
    parser.add_argument("--input-path", type=str, default="", help="path to input model")
    parser.add_argument("--output-path", type=str, default="", help="path to output model")
    parser.add_argument("--device", default=-1, type=int, help="device number")
    parser.add_argument("--hardware", default="310", help="310 or 910")
    parser.add_argument("--handle-type", type=str, required=True,
                        choices=["cut_quant", "cut_float"])
    args = parser.parse_args()

    return args

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.input_path, use_fast=False)
    model = TelechatForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.float32).cpu()
    return model, tokenizer

def cut_weights_float(state_dict, world_size, cut_row_keys=("query", "key_value", "gate_proj", "up_proj"), cut_col_keys=("dense", "down_proj")):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in state_dict.items():
        key_short = key.split(".")[-2]
        key_type = key.split(".")[-1]
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            if key_type == "weight":
                cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
            elif key_type == "bias":
                # 浮点加速库bias横切法
                cut_tensor_list = [tensor] * world_size
        else:
            cut_tensor_list = [tensor] * world_size
        
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_model_float(args):
    device = torch.device(f"npu:{args.device}" if args.device > 0 else "cpu")
    model, tokenizer = load_model(args)
    model = model.half().to(device)

    tokenizer.save_pretrained(f"{args.output_path}/tokenizer")

    state_dict_list = cut_weights_float(model.state_dict(), args.world_size)

    model_config = model.config
    model_config.world_size = args.world_size
    create_model = TelechatForCausalLM(model_config).half().to(device)
    for i in range(args.world_size):
        create_model.load_state_dict(state_dict_list[i])
        create_model.save_pretrained(f"{args.output_path}/part_model/{i}/", max_shard_size="4096MB")
    print(f"save successfully to: {args.output_path}")


def process_deq_scale(deq_scale_dict):
    new_deq_scale_dict = {}
    for key, deq_scale in deq_scale_dict.items():
        new_deq_scale = np.frombuffer(deq_scale.numpy().tobytes(), dtype=np.int32)
        new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
    return new_deq_scale_dict


def bias_correction_new(fp_bias, quant_weight, input_offset, deq_scale):
    bias_correction = fp_bias.npu() / deq_scale.npu() - quant_weight.to(torch.float32).npu().sum(dim=1) * float(input_offset)
    return bias_correction


def cut_weights_quant(state_dict, world_size, cut_row_keys=("query", "key_value", "gate_proj", "up_proj"), cut_col_keys=("dense", "down_proj")):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in state_dict.items():
        key_short = key.split(".")[-1]
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        else:
            cut_tensor_list = [tensor] * world_size
        
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_bias_quant(state_dict, world_size, is_bias=False, cut_row_keys=("query", "key_value", "gate_proj", "up_proj"), cut_col_keys=("dense", "down_proj")):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in state_dict.items():
        key_short = key.split(".")[-1]
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        else:
            if is_bias:
                tensor = tensor / world_size
            cut_tensor_list = [tensor] * world_size
        
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_model_quant(args):
    weight_path = args.input_path

    print(f"loading quant weight from {weight_path}")
    quant_weight_dict = np.load(f"{weight_path}/quant_weight.npy", allow_pickle=True).item() 
    deq_scale_dict = np.load(f"{weight_path}/deq_scale.npy", allow_pickle=True).item() 
    fp_bias_dict = np.load(f"{weight_path}/fp_bias.npy", allow_pickle=True).item() 
    quant_bias_dict = np.load(f"{weight_path}/quant_bias.npy", allow_pickle=True).item() 
    input_offset_dict = np.load(f"{weight_path}/input_offset.npy", allow_pickle=True).item() 

    print("correcting bias...")
    bias = {}
    for k in fp_bias_dict.keys():
        bias[k] = bias_correction_new(fp_bias_dict[k], quant_weight_dict[k], input_offset_dict[k], deq_scale_dict[k])
    np.save(os.path.join(weight_path, "bias.npy"), bias)
    print(f"corrected bias saved to {weight_path}")

    new_deq_scale_dict = process_deq_scale(deq_scale_dict)
    np.save(os.path.join(weight_path, "new_deq_scale.npy"), new_deq_scale_dict)

    state_quant_weight_dict_list = cut_weights_quant(quant_weight_dict, args.world_size)
    state_fp_bias_dict_list = cut_bias_quant(bias, args.world_size, True)
    state_deq_scale_dict_list = cut_bias_quant(new_deq_scale_dict, args.world_size)

    save_path = os.path.join(args.output_path, "part_model")
    print(f"saving model to {save_path}...")
    for i in range(args.world_size):
        base_path = os.path.join(save_path, str(i))
        os.makedirs(base_path, exist_ok=True)
        np.save(os.path.join(base_path, "quant_weight.npy"), state_quant_weight_dict_list[i])
        np.save(os.path.join(base_path, "fp_bias.npy"), state_fp_bias_dict_list[i])
        np.save(os.path.join(base_path, "deq_scale.npy"), state_deq_scale_dict_list[i])

        shutil.copyfile(os.path.join(weight_path, "input_offset.npy"), os.path.join(base_path, "input_offset.npy"))
        shutil.copyfile(os.path.join(weight_path, "weight_offset.npy"), os.path.join(base_path, "weight_offset.npy"))
        shutil.copyfile(os.path.join(weight_path, "input_scale.npy"), os.path.join(base_path, "input_scale.npy"))
        shutil.copyfile(os.path.join(weight_path, "weight_scale.npy"), os.path.join(base_path, "weight_scale.npy"))
    print(f"save successfully to: {args.output_path}")



if __name__ == "__main__":
    args = get_args()
    if args.handle_type == "cut_float":
        cut_model_float(args)
    elif args.handle_type == "cut_quant":
        cut_model_quant(args)
    else:
        raise Exception("handle_type invalid!")
