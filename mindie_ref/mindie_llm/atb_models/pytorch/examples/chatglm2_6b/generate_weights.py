# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2031. All rights reserved

import os
import shutil
import argparse


import torch
import torch_npu
import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Cut Model weights.")
    parser.add_argument("--model_path",
                        required=True,
                        help="Location of Model weights, which contains model folders")
    parser.add_argument("--parallel_float_weight_path",
                        default='tensor_parallel',
                        help="Location to write the part weights")
    parser.add_argument("--tp_size", type=int, default=1, help="the size of  parallel")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--model_file_golden",
                        default="patches/modeling_chatglm_npu_parallel.py",
                        help="model_file_golden")
    args = parser.parse_args()
    return args


def bias_correction(fp_bias_dict, quant_weight_dict, input_offset_dict, deq_scale_dict):
    new_bias_dict = {}
    for key in fp_bias_dict.keys():
        new_bias_dict[key] = fp_bias_dict[key].npu()/deq_scale_dict[key].npu() - quant_weight_dict[key].to(torch.float32).npu().sum(dim=1) * float(input_offset_dict[key])
        new_bias_dict[key] = new_bias_dict[key].detach().to(torch.int32)
    return new_bias_dict


def deq_scale_process(deq_scale_dict):
    new_deq_scale_dict = {}
    for key, deq_scale in deq_scale_dict.items():
        deq_scale = deq_scale.numpy()
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
        new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
    return new_deq_scale_dict


def cut_fp_tensors(model_cfg, key, tensor, tp_size):
    if tp_size == 1:
        return [tensor]

    cut_tensor_list = []
    if 'dense_h_to_4h' in key: # weight
        chunk_tensors = torch.chunk(tensor, tp_size * 2, dim=0)
        cut_tensor_list = [torch.cat([chunk_tensors[i], chunk_tensors[i+tp_size]], dim=0)
                            for i in range(tp_size)]
    elif 'dense_4h_to_h' in key or 'self_attention.dense.weight' in key: # weight x 2
        cut_tensor_list = torch.chunk(tensor, tp_size, dim=1)
    elif 'query_key_value' in key: # weight and bias
        hidden_size_per_attention_head = model_cfg.hidden_size // model_cfg.num_attention_heads
        num_attention_heads_per_partition = model_cfg.num_attention_heads
        num_multi_query_groups_per_partition = model_cfg.multi_query_group_num
        query_layer, key_layer, value_layer = tensor.split(
            [
                hidden_size_per_attention_head * num_attention_heads_per_partition,
                hidden_size_per_attention_head * num_multi_query_groups_per_partition,
                hidden_size_per_attention_head * num_multi_query_groups_per_partition
            ],
            dim=0
        )
        kv_tp_size = min(tp_size, num_multi_query_groups_per_partition)
        query_list = torch.chunk(query_layer, tp_size, dim=0)
        key_list = torch.chunk(key_layer, kv_tp_size, dim=0)
        value_list = torch.chunk(value_layer, kv_tp_size, dim=0)
        cut_tensor_list = [torch.cat([query_list[i], key_list[i*kv_tp_size//tp_size], value_list[i*kv_tp_size//tp_size]], dim=0)
                            for i in range(tp_size)]
    else:
        cut_tensor_list = [tensor] * tp_size
    return cut_tensor_list


def cut_quant_tensors(model_cfg, key, tensor, tp_size, mode):
    if tp_size == 1:
        return [tensor]
    
    cut_tensor_list = []
    if 'dense_h_to_4h' in key: 
        chunk_tensors = torch.chunk(tensor, tp_size * 2, dim=0)
        cut_tensor_list = [torch.cat([chunk_tensors[i], chunk_tensors[i+tp_size]], dim=0)
                            for i in range(tp_size)]
    elif ('dense_4h_to_h' in key or 'self_attention.dense' in key):
        if mode == "weights":
            cut_tensor_list = torch.chunk(tensor, tp_size, dim=1) 
        elif mode == "bias":
            zero_tensor = torch.zeros(tensor.shape, dtype=tensor.dtype)
            cut_tensor_list = [tensor] + [zero_tensor] * (tp_size - 1)
        else:
            cut_tensor_list = [tensor] * tp_size
    elif 'query_key_value' in key: 
        hidden_size_per_attention_head = model_cfg.hidden_size // model_cfg.num_attention_heads
        num_attention_heads_per_partition = model_cfg.num_attention_heads
        num_multi_query_groups_per_partition = model_cfg.multi_query_group_num
        query_layer, key_layer, value_layer = tensor.split(
            [
                hidden_size_per_attention_head * num_attention_heads_per_partition,
                hidden_size_per_attention_head * num_multi_query_groups_per_partition,
                hidden_size_per_attention_head * num_multi_query_groups_per_partition
            ],
            dim=0
        )
        kv_tp_size = min(tp_size, num_multi_query_groups_per_partition)
        query_list = torch.chunk(query_layer, tp_size, dim=0)
        key_list = torch.chunk(key_layer, kv_tp_size, dim=0)
        value_list = torch.chunk(value_layer, kv_tp_size, dim=0)
        cut_tensor_list = [torch.cat([query_list[i], key_list[i*kv_tp_size//tp_size], value_list[i*kv_tp_size//tp_size]], dim=0)
                            for i in range(tp_size)]
    else:
        cut_tensor_list = [tensor] * tp_size
    return cut_tensor_list


def cut_fp_weights(model_inst, tp_size):
    model_cfg = model_inst.config
    state_dict_list = [{} for i in range(tp_size)]
    for key, tensor in model_inst.state_dict().items():
        cut_tensor_list = cut_fp_tensors(model_cfg, key, tensor, tp_size)
        for i in range(tp_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_fp_1st_layer_weights(model_inst, tp_size):
    model_cfg = model_inst.config
    state_dict_list = [{} for i in range(tp_size)]
    for key, tensor in model_inst.state_dict().items():
        key_short = ".".join([key.split(".")[-2], key.split(".")[-1]])
        if '.0.' in key:
            cut_tensor_list = cut_fp_tensors(model_cfg, key, tensor, tp_size)
            for i in range(tp_size):
                state_dict_list[i][key_short] = cut_tensor_list[i]
    return state_dict_list


def cut_quant_weights(model_cfg, weights, tp_size):
    state_dict_list = [{} for i in range(tp_size)]
    for key, tensor in weights.items():
        cut_tensor_list = cut_quant_tensors(model_cfg, key, tensor, tp_size, mode="weights")
        for i in range(tp_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_quant_bias(model_cfg, bias, tp_size):
    state_dict_list = [{} for i in range(tp_size)]
    for key, tensor in bias.items():
        cut_tensor_list = cut_quant_tensors(model_cfg, key, tensor, tp_size, mode="bias")
        for i in range(tp_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_quant_scales(model_cfg, bias, tp_size):
    state_dict_list = [{} for i in range(tp_size)]
    for key, tensor in bias.items():
        cut_tensor_list = cut_quant_tensors(model_cfg, key, tensor, tp_size, mode="scales")
        for i in range(tp_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


if __name__ == "__main__":
    
    # git options and configs
    args = parse_args()
    ENABLE_QUANT = os.environ.get("ENABLE_QUANT", "0") == "1"
    ENABLE_SPARSE = os.environ.get("ENABLE_SPARSE", "0") == "1"
    torch.npu.set_device(args.device)

    # load original model and cut float weights
    shutil.copy(args.model_file_golden, os.path.join(args.model_path, "modeling_chatglm.py"))
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half()
    
    model_config = model.config
    model_config.world_size = args.tp_size

    # save new model config and parallel float weights
    if args.tp_size > 1:
        save_path = os.path.join(args.model_path, args.parallel_float_weight_path)
        if os.path.exists(save_path):
            print(f"[info]: The parallel float weights has exist in '{save_path}'.")
        else:
            state_dict_list = cut_fp_weights(model, args.tp_size)
            parallel_model = AutoModel.from_config(model_config, trust_remote_code=True)
            for i in range(args.tp_size):
                target_dir = os.path.join(args.model_path, args.parallel_float_weight_path , "part_model", str(i))
                parallel_model.load_state_dict(state_dict_list[i])
                parallel_model.save_pretrained(target_dir)
                for source_file in ["configuration_chatglm.py", "quantization.py"]:
                    shutil.copy(os.path.join(args.model_path, source_file), target_dir)
            print(f"[info]: The parallel float weights has been saved to '{save_path}'.")
    
    # cut or correct quant weights
    if ENABLE_QUANT or ENABLE_SPARSE:
        # load quant weights
        QUANT_WEIGHT_PATH = os.environ.get("QUANT_WEIGHT_PATH")
        input_offset_dict = np.load(os.path.join(QUANT_WEIGHT_PATH, "input_offset.npy"), allow_pickle=True).item()
        quant_weight_dict = np.load(os.path.join(QUANT_WEIGHT_PATH, "quant_weight.npy"), allow_pickle=True).item()
        deq_scale_dict = np.load(os.path.join(QUANT_WEIGHT_PATH, "deq_scale.npy"), allow_pickle=True).item()
        fp_bias_dict = np.load(os.path.join(QUANT_WEIGHT_PATH, "fp_bias.npy"), allow_pickle=True).item()
        
        # correct bias and deq_scale
        bias_dict = bias_correction(fp_bias_dict, quant_weight_dict, input_offset_dict, deq_scale_dict)
        deq_scale_dict = deq_scale_process(deq_scale_dict)

        # handle float weights first layer
        state_quant_weight_dict_list = cut_fp_1st_layer_weights(model, args.tp_size) 

        # handle quant weights of other layers
        for i in range(args.tp_size):
            np.save(os.path.join(QUANT_WEIGHT_PATH, f"weight{i}.npy"), state_quant_weight_dict_list[i])
            np.save(os.path.join(QUANT_WEIGHT_PATH, f"bias{i}.npy"), state_quant_weight_dict_list[i])
        print("[info]: save float weights of 1st layer")
        
        state_quant_weight_dict_list = cut_quant_weights(model_config, quant_weight_dict, args.tp_size)
        state_bias_dict_list = cut_quant_bias(model_config, bias_dict, args.tp_size)
        state_deq_scale_dict_list = cut_quant_scales(model_config, deq_scale_dict, args.tp_size)
        for i in range(args.tp_size):
            np.save(os.path.join(QUANT_WEIGHT_PATH, f"quant_weight{i}.npy"), state_quant_weight_dict_list[i])
            np.save(os.path.join(QUANT_WEIGHT_PATH, f"new_bias{i}.npy"), state_bias_dict_list[i])
            np.save(os.path.join(QUANT_WEIGHT_PATH, f"new_deq_scale{i}.npy"), state_deq_scale_dict_list[i])
        print("[info]: save quant weights of the other layers")
        print(f"[info]: The processed quant weights has been saved to '{QUANT_WEIGHT_PATH}'.")