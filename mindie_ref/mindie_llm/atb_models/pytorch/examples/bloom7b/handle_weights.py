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
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, BloomTokenizerFast, AutoConfig

from modeling_bloom_cut import BloomParallelForCausalLM


def get_args():
    parser = argparse.ArgumentParser(description="Bloom info.")
    parser.add_argument("--input-path", default="./model/bloom/", help="input model path",)
    parser.add_argument("--output-path", default='./', help="output model path")
    parser.add_argument("--device", default=-1, type=int, help="device number")
    parser.add_argument(
        "--handle-type", type=str, required=True,
        choices=[
            'quant',  # 量化权重
            'cut_quant',  # 切分量化权重
            'cut_float'  # 切分浮点权重
            ])
    p_args = parser.parse_args()
    return p_args


def load_model(in_args):
    tokenizer = AutoTokenizer.from_pretrained(in_args.input_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(in_args.input_path, torch_dtype=torch.float32).cpu()
    return model, tokenizer


def get_calib_dataset(tokenizer):
    calib_list = [
        "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who was the first president of the United States\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the name of the vice president of the United States\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
        ]
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors="pt", max_length=32, padding='max_length', truncation=True)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'].cpu(), None, inputs.data['attention_mask'].cpu()])
    return calib_dataset


def bias_correction(fp_bias, quant_weight, input_offset, deq_scale, device_type='310'):
    bias_out = fp_bias / deq_scale - quant_weight.to(torch.float32).sum(dim=1) * float(input_offset)
    return bias_out


def process_deq_scale(deq_scale_dict):
    new_deq_scale_dict = {}
    for key, deq_scale in deq_scale_dict.items():
        deq_scale = deq_scale.numpy()
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.int32)
        new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
    return new_deq_scale_dict


def quant_model(args_quant, verbose=False):
    from llm_ptq_tools import Calibrator, QuantConfig

    print("loading model...")
    model, tokenizer = load_model(args_quant)

    os.makedirs(args_quant.output_path, exist_ok=True)

    tokenizer.save_pretrained(args_quant.output_path)
    model.config.to_json_file(os.path.join(args_quant.output_path, "config.json"))

    # 保存非量化的部分权重
    float_layer_ids = [7]
    saved_float_keys = []
    weights_keys = model.state_dict().keys()
    for weights_key in weights_keys:
        key_split = weights_key.split('.')
        is_split_layer = 'input_layernorm' in key_split or 'post_attention_layernorm' in key_split
        if 'h' in key_split and (int(key_split[2]) in float_layer_ids or is_split_layer):
            saved_float_keys.append(weights_key)
        elif "h" not in key_split:
            saved_float_keys.append(weights_key)
    saved_float_weights = {key: model.state_dict()[key] for key in saved_float_keys}
    torch.save(saved_float_weights, os.path.join(args_quant.output_path, "float_layers_weights.pt"))

    print("model loaded, starting quant model...")
    dataset_calib = get_calib_dataset(tokenizer)
    quant_config = QuantConfig(w_bit=8, disable_names=[], dev_type='cpu', act_method=3, pr=1.0, mm_tensor=False, w_hessian=False)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L1')
    calibrator.run()

    if verbose:
        for item in dataset_calib:
            with torch.no_grad():
                output = model.generate(
                    item[0],
                    max_new_tokens=32,
                    attention_mask=item[2],
                    use_cache=True,
                )
                res = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                print(res)
    
    print("starting saving model...")
    calibrator.save(args_quant.output_path)

    print("quant model done, correcting bias...")

    input_offset_dict = np.load(os.path.join(args_quant.output_path, "input_offset.npy"), allow_pickle=True).item()
    quant_weight_dict = np.load(os.path.join(args_quant.output_path, "quant_weight.npy"), allow_pickle=True).item()
    deq_scale_dict = np.load(os.path.join(args_quant.output_path, "deq_scale.npy"), allow_pickle=True).item()
    fp_bias_dict = np.load(os.path.join(args_quant.output_path, "fp_bias.npy"), allow_pickle=True).item()

    bias = {}
    for i in fp_bias_dict.keys():
        bias[i] = bias_correction(fp_bias_dict[i], 
                                quant_weight_dict[i], 
                                int(input_offset_dict[i]), 
                                deq_scale_dict[i]).cpu()
    print("correcting deq_scale...")
    new_deq_scale_dict = process_deq_scale(deq_scale_dict)
    np.save(os.path.join(args_quant.output_path, "bias.npy"), bias)
    # 覆写旧的deq_scale
    np.save(os.path.join(args_quant.output_path, "deq_scale.npy"), new_deq_scale_dict)
    print("all done!")


def cut_weights_float(state_dict, world_size, cut_row_keys=('dense_h_to_4h'), cut_col_keys=('dense', 'dense_4h_to_h')):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in state_dict.items():
        key_short = key.split('.')[-2]
        key_type = key.split('.')[-1]

        if key_short == 'query_key_value':
            num_heads = 32
            head_dim = 128
            if key_type == "weight":
                tensor = tensor.view(num_heads, 3, head_dim, num_heads * head_dim)
            elif key_type == "bias":
                tensor = tensor.view(num_heads, 3, head_dim)
            tensor_list = (tensor[:, 0, ...], tensor[:, 1, ...], tensor[:, 2, ...])
            cut_tensor_list = [torch.Tensor([]).half(), torch.Tensor([]).half()]
            for i in range(3):
                cut_tensor_list[0] = torch.cat(
                    (cut_tensor_list[0], torch.chunk(tensor_list[i], world_size, dim=0)[0]), 1)
                cut_tensor_list[1] = torch.cat(
                    (cut_tensor_list[1], torch.chunk(tensor_list[i], world_size, dim=0)[1]), 1)
            if key_type == "weight":
                cut_tensor_list[0] =  cut_tensor_list[0].reshape(num_heads * head_dim * 3 // 2, num_heads * head_dim)
                cut_tensor_list[1] =  cut_tensor_list[1].reshape(num_heads * head_dim * 3 // 2, num_heads * head_dim)
            elif key_type == "bias":
                cut_tensor_list[0] =  cut_tensor_list[0].reshape(num_heads * head_dim * 3 // 2)
                cut_tensor_list[1] =  cut_tensor_list[1].reshape(num_heads * head_dim * 3 // 2)
        else:
            if key_short in cut_row_keys:
                cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
            elif key_short in cut_col_keys:
                if key_type == "weight":
                    cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
                elif key_type == "bias":
                    # tensor = tensor / 2
                    cut_tensor_list = [tensor] * world_size
            else:
                cut_tensor_list = [tensor] * world_size

        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_model_float(args_float):
    device = torch.device(f"npu:{args_float.device}" if args_float.device > 0 else "cpu")
    model, tokenizer = load_model(args_float)
    model = model.half().to(device)
    
    tokenizer.save_pretrained(args_float.output_path + '/tokenizer')

    state_dict_list = cut_weights_float(model.state_dict(), args_float.world_size)
    model_config = model.config
    model_config.world_size = args_float.world_size
    create_model = BloomParallelForCausalLM(model_config).half().to(device)
    for i in range(args_float.world_size):
        create_model.load_state_dict(state_dict_list[i])
        create_model.save_pretrained(os.path.join(args_float.output_path, 'part_model', str(i)),
                                     max_shard_size = "4096MB")
    print('Tensor parallelism weights have been successfully saved.')


def cut_weights_quant(weight, world_size):
    state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in weight.items():
        cut_tensor_list = []
        key_short = ".".join([key.split(".")[-2], key.split(".")[-1]]) 
        if key_short in ["mlp.dense_h_to_4h"]:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in ["mlp.dense_4h_to_h", "self_attention.dense"]:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        elif key_short in ["self_attention.query_key_value"]:
            # model.config.n_head
            num_heads = 32
            head_dim = 4096 // num_heads
            tensor = tensor.view(num_heads, 3, head_dim, num_heads * head_dim)            
            tensor_list = (tensor[:, 0, ...], tensor[:, 1, ...], tensor[:, 2, ...])
            cut_tensor_list = [torch.Tensor([]) for _ in range(world_size)]
            for i in range(3):
                for j in range(world_size):
                    cut_tensor_list[j] = torch.cat((cut_tensor_list[j], torch.chunk(tensor_list[i], world_size, dim=0)[j]), 1)
            for j in range(world_size):
                cut_tensor_list[j] = cut_tensor_list[j].reshape(num_heads * head_dim * 3 // world_size, num_heads * head_dim)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_bias_quant(bias, world_size, is_bias=False):

    state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in bias.items():
        # cut tensors
        cut_tensor_list = []
        key_short = ".".join([key.split(".")[-2], key.split(".")[-1]])
        if key_short in ["mlp.dense_h_to_4h"]:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in ["self_attention.query_key_value"]:
            num_heads = 32 # model.config.n_head
            head_dim = 4096 // num_heads
            tensor = tensor.view(num_heads, 3, head_dim)
            tensor_list = (tensor[:, 0, ...], tensor[:, 1, ...], tensor[:, 2, ...])
            cut_tensor_list = [torch.Tensor([]) for _ in range(world_size)]
            for i in range(3):
                for j in range(world_size):
                    cut_tensor_list[j] = torch.cat((cut_tensor_list[j], torch.chunk(tensor_list[i], world_size, dim=0)[j]), 1)
            for j in range(world_size):
                cut_tensor_list[j] = cut_tensor_list[j].reshape(num_heads * head_dim * 3 // world_size)
        else:
            if is_bias:
                tensor = tensor / 2.0
            cut_tensor_list = [tensor] * world_size
        # # assign state_dict_list
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]

    return state_dict_list


def cut_model_quant(args_quant):
    weight_path = args_quant.input_path

    tokenizer = BloomTokenizerFast.from_pretrained(args_quant.input_path, use_fast=False)
    tokenizer.save_pretrained(os.path.join(args_quant.output_path, 'tokenizer'))

    config = AutoConfig.from_pretrained(args_quant.input_path)
    print(f"=========quant weight path:{weight_path} ==========")
    quant_weight_dict = np.load(os.path.join(weight_path, "quant_weight.npy"), allow_pickle=True).item()
    deq_scale_dict = np.load(os.path.join(weight_path, "deq_scale.npy"), allow_pickle=True).item()
    bias_dict = np.load(os.path.join(weight_path, "bias.npy"), allow_pickle=True).item()

    float_weight_dict = torch.load(os.path.join(weight_path, "float_layers_weights.pt"))

    state_quant_weight_dict_list = cut_weights_quant(quant_weight_dict, args_quant.world_size)
    state_bias_dict_list = cut_bias_quant(bias_dict, args_quant.world_size, True)
    state_deq_scale_dict_list = cut_bias_quant(deq_scale_dict, args_quant.world_size)

    float_weight_dict_list = cut_weights_float(float_weight_dict, args_quant.world_size)

    save_path = os.path.join(args_quant.output_path, 'part_model')
    print(f"=========part quant weight path:{save_path} ==========")
    for i in range(args_quant.world_size):
        base_path = os.path.join(save_path, str(i))
        os.makedirs(base_path, exist_ok=True)
        config.to_json_file(os.path.join(base_path, "config.json"))
        np.save(os.path.join(base_path, "quant_weight.npy"), state_quant_weight_dict_list[i])
        np.save(os.path.join(base_path, "bias.npy"), state_bias_dict_list[i])
        np.save(os.path.join(base_path, "deq_scale.npy"), state_deq_scale_dict_list[i])

        torch.save(float_weight_dict_list[i], os.path.join(base_path, "float_layers_weights.pt"))

        shutil.copyfile(os.path.join(weight_path, "input_offset.npy"), os.path.join(base_path, "input_offset.npy"))
        shutil.copyfile(os.path.join(weight_path, "input_scale.npy"), os.path.join(base_path, "input_scale.npy"))

    print('Tensor parallelism weights have been successfully saved.')
    print("the location of parallel quant weight is {}".format(save_path))


if __name__ == "__main__":
    args = get_args()
    args.world_size = 2
    if args.handle_type == "quant":
        quant_model(args)
    elif args.handle_type == "cut_quant":
        cut_model_quant(args)
    elif args.handle_type == "cut_float":
        cut_model_float(args)
    else:
        raise Exception("handle_type error!")