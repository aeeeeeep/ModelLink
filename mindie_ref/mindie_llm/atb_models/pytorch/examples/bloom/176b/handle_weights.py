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
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomTokenizerFast, AutoConfig

from modeling_bloom_ascend import BloomCommonForCausalLM as BloomForCausalLM


class CutWeightsConfig:
    def __init__(self, state_dict, world_size, config):
        self.state_dict = state_dict
        self.world_size = world_size
        self.config = config
        self.recuce_bias = False
        self.cut_row_keys = ('dense_h_to_4h',)
        self.cut_col_keys = ('dense', 'dense_4h_to_h', 'word_embeddings')


def get_args():
    parser = argparse.ArgumentParser(description="Bloom info.")
    parser.add_argument("--input-path", default="./model/bloom/", help="input model path",)
    parser.add_argument("--output-path", default='./', help="output model path")
    parser.add_argument("--device", default=-1, type=int, help="device number")
    parser.add_argument("--world-size", default=8, type=int, help="world size")
    parser.add_argument(
        "--handle-type", type=str, required=True,
        choices=[
            'quant',      # 量化权重
            'cut_quant',  # 切分量化权重
            'cut_float'   # 切分浮点权重
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


def quant_model(args_quant, verbose=False):
    if not os.environ.get("ASCEND_TOOLKIT_HOME"):
        raise Exception("Environment variable ASCEND_TOOLKIT_HOME not found. Please source /path/to/cann/set_env.sh")
    
    modelslim_path = os.path.join(os.environ.get("ASCEND_TOOLKIT_HOME"), "tools")
    import sys
    sys.path = [modelslim_path] + sys.path
    from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

    print("[~] loading model...")
    t0 = time.time()
    model, tokenizer = load_model(args_quant)
    print(f"[+] load model: {time.time() - t0:.1f}s")
    os.makedirs(args_quant.output_path, exist_ok=True)

    tokenizer.save_pretrained(args_quant.output_path)
    model.config.to_json_file(os.path.join(args_quant.output_path, "config.json"))

    for name, module in model.named_modules():
        print(name, module)

    # 保存非量化的部分权重
    float_layer_ids = []
    saved_float_keys = []
    weights_keys = model.state_dict().keys()
    for weights_key in weights_keys:
        print(weights_key, "." * (80 - len(str(weights_key))), model.state_dict()[weights_key].shape)

        key_split = weights_key.split('.')
        is_split_layer = any(_n in key_split for _n in ('input_layernorm', 'post_attention_layernorm', 'bias'))
        if 'h' in key_split and (int(key_split[2]) in float_layer_ids or is_split_layer):
            saved_float_keys.append(weights_key)
        elif "h" not in key_split:
            saved_float_keys.append(weights_key)
    saved_float_weights = {key: model.state_dict()[key] for key in saved_float_keys}
    torch.save(saved_float_weights, os.path.join(args_quant.output_path, "float_layers_weights.pt"))

    print("model loaded, starting quant model...")
    dataset_calib = get_calib_dataset(tokenizer)
    quant_config = QuantConfig(w_bit=8, a_bit=16, disable_names=[], dev_type='cpu', act_method=3, pr=1.0, mm_tensor=False)
    calibrator = Calibrator(model, quant_config, calib_data=None, disable_level='L0')
    calibrator.run()
    
    print("starting saving model...")
    calibrator.save(args_quant.output_path)

    quant_weight_dict = np.load(os.path.join(args_quant.output_path, "quant_weight.npy"), allow_pickle=True).item()
    if "lm_head" in quant_weight_dict:
        del quant_weight_dict["lm_head"]
    np.save(os.path.join(args_quant.output_path, "quant_weight.npy"), quant_weight_dict)


def cut_weights(cfg):
    state_dict_list = [{} for i in range(cfg.world_size)]
    for key, tensor in cfg.state_dict.items():
        if "lm_head" in key:
            continue

        key_short = key.split('.')[-2]
        key_type = key.split('.')[-1]

        if key_short == 'query_key_value':
            num_heads, head_dim = cfg.config.n_head, cfg.config.hidden_size // cfg.config.n_head
            dst_shape = list(tensor.shape)
            dst_shape[0] //= cfg.world_size

            tensor = tensor.view(num_heads, 3, head_dim, -1)
            tensor_list = torch.unbind(tensor, dim=1)
            chunk_tensor_list = [torch.chunk(item, cfg.world_size, dim=0) for item in tensor_list]
            cut_tensor_list = [torch.cat(item, 1).reshape(*dst_shape) for item in zip(*chunk_tensor_list)]
        else:
            if key_short in cfg.cut_row_keys:
                cut_tensor_list = torch.chunk(tensor, cfg.world_size, dim=0)
            elif key_short in cfg.cut_col_keys:
                if key_type == "weight":
                    cut_tensor_list = torch.chunk(tensor, cfg.world_size, dim=1)
                elif key_type == "bias":
                    cut_tensor_list = [tensor] * cfg.world_size
            else:
                cut_tensor_list = [tensor] * cfg.world_size

        for i in range(cfg.world_size):
            state_dict_list[i][key] = torch.clone(cut_tensor_list[i].contiguous())
    return state_dict_list


def cut_weights_quant(weight_type, state_dict, world_size, config, recuce_bias=False):
    state_dict = {k + "." + weight_type: v for k, v in state_dict.items()}
    cut_weights_config = CutWeightsConfig(state_dict, world_size, config)
    cut_weights_config.recuce_bias = recuce_bias
    state_dict_list = cut_weights(cut_weights_config)
    state_dict_list = [{k.rstrip(weight_type)[:-1]: v for k, v in state_dict_tmp.items()} for state_dict_tmp in state_dict_list]
    return state_dict_list


def cut_model_float(args_float):
    device = torch.device(f"npu:{args_float.device}" if args_float.device > 0 else "cpu")
    model, tokenizer = load_model(args_float)
    model = model.half().to(device)
    
    tokenizer.save_pretrained(os.path.join(args_float.output_path, 'tokenizer'))
    cut_weights_config = CutWeightsConfig(model.state_dict(), args_float.world_size, model.config)
    state_dict_list = cut_weights(cut_weights_config)
    model_config = model.config
    model_config.world_size = args_float.world_size
    create_model = BloomForCausalLM(model_config).half().to(device)
    for i in range(args_float.world_size):
        create_model.load_state_dict(state_dict_list[i])
        create_model.save_pretrained(os.path.join(args_float.output_path, 'part_model', str(i)),
                                     max_shard_size="2048MB")
    print("Tensor parallelism weights have been successfully saved.")


def cut_model_quant(args_quant):
    weight_path = args_quant.input_path

    tokenizer = BloomTokenizerFast.from_pretrained(args_quant.input_path, use_fast=False)
    tokenizer.save_pretrained(os.path.join(args_quant.output_path, 'tokenizer'))

    config = AutoConfig.from_pretrained(args_quant.input_path)
    print(f"=========quant weight path:{weight_path} ==========")
    quant_weight_dict = np.load(os.path.join(weight_path, "quant_weight.npy"), allow_pickle=True).item()
    weight_scale_dict = np.load(os.path.join(weight_path, "weight_scale.npy"), allow_pickle=True).item()
    float_weight_dict = torch.load(os.path.join(weight_path, "float_layers_weights.pt"))

    state_quant_weight_dict_list = cut_weights_quant("weight", quant_weight_dict, args_quant.world_size, config)
    state_weight_scale_dict_list = cut_weights_quant("bias", weight_scale_dict, args_quant.world_size, config)
    cut_weights_config = CutWeightsConfig(float_weight_dict, args_quant.world_size, config)
    float_weight_dict_list = cut_weights(cut_weights_config)

    save_path = os.path.join(args_quant.output_path, 'part_model')
    print(f"=========part quant weight path:{save_path} ==========")
    for i in range(args_quant.world_size):
        base_path = os.path.join(save_path, str(i))
        os.makedirs(base_path, exist_ok=True)
        config.to_json_file(os.path.join(base_path, "config.json"))
        np.save(os.path.join(base_path, "quant_weight.npy"), state_quant_weight_dict_list[i])
        np.save(os.path.join(base_path, "weight_scale.npy"), state_weight_scale_dict_list[i])
        torch.save(float_weight_dict_list[i], os.path.join(base_path, "float_layers_weights.pt"))

    print("Tensor parallelism weights have been successfully saved.")
    print("the location of parallel quant weight is {}".format(save_path))


if __name__ == "__main__":
    main_args = get_args()
    if main_args.handle_type == "quant":
        quant_model(main_args)
    elif main_args.handle_type == "cut_quant":
        cut_model_quant(main_args)
    elif main_args.handle_type == "cut_float":
        cut_model_float(main_args)
    else:
        raise Exception("handle_type error!")
