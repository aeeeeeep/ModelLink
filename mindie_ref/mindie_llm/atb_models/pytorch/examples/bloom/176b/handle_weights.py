import argparse
import os
import json
import time
import tqdm
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomTokenizerFast, AutoConfig

from modeling_bloom_ascend import BloomCommonForCausalLM as BloomForCausalLM

from safetensors import safe_open
from safetensors.torch import save_file


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


def quant_model(args_quant):
    if not os.environ.get("ASCEND_TOOLKIT_HOME"):
        raise Exception("Environment variable ASCEND_TOOLKIT_HOME not found. Please source /path/to/cann/set_env.sh")
    
    modelslim_path = os.path.join(os.environ.get("ASCEND_TOOLKIT_HOME"), "tools")
    import sys
    sys.path = [modelslim_path] + sys.path
    from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

    print("[~] loading model...")
    t0 = time.time()
    model, tokenizer = load_model(args_quant)
    print(f"[+] load model: {time.time() - t0:.1f}s")  # load model: 1792.4s
    os.makedirs(args_quant.output_path, exist_ok=True)

    tokenizer.save_pretrained(args_quant.output_path)
    model.config.to_json_file(os.path.join(args_quant.output_path, "config.json"))
    print("[~] model loaded, starting quant model...")

    quant_config = QuantConfig(w_bit=8, a_bit=16, disable_names=[], dev_type='cpu', act_method=3, pr=1.0, mm_tensor=False)
    calibrator = Calibrator(model, quant_config, calib_data=None, disable_level='L0')
    calibrator.run()
    print("[~] starting saving model...")
    calibrator.save(args_quant.output_path, save_type=["safe_tensor"])


def cut_weights(cfg):
    state_dict_list = [{} for _ in range(cfg.world_size)]
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
        elif key_short in cfg.cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, cfg.world_size, dim=0)
        elif key_short in cfg.cut_col_keys:
            if key_type == "weight":
                cut_tensor_list = torch.chunk(tensor, cfg.world_size, dim=1)
            elif key_type == "bias":
                cut_tensor_list = [tensor] * cfg.world_size
            else:
                cut_tensor_list = [tensor] * cfg.world_size
        else:
            cut_tensor_list = [tensor] * cfg.world_size

        for i in range(cfg.world_size):
            state_dict_list[i][key] = torch.clone(cut_tensor_list[i].contiguous())
    return state_dict_list


def cut_weights_quant(state_dict, world_size, config):
    state_dict = {k: v for k, v in state_dict.items()}
    cut_weights_config = CutWeightsConfig(state_dict, world_size, config)
    state_dict_list = cut_weights(cut_weights_config)
    state_dict_list = [{k: v for k, v in state_dict_tmp.items()} for state_dict_tmp in state_dict_list]
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
    print("================================================================================")
    # The path to save the weights.
    weight_path = args_quant.input_path
    print(f"[+] quant weight dir: {weight_path}")

    # Save the tokenizer to the output path.
    tokenizer = BloomTokenizerFast.from_pretrained(args_quant.input_path, use_fast=False)
    tokenizer.save_pretrained(os.path.join(args_quant.output_path, 'tokenizer'))
    print(f"[+] The tokenizer has been saved successfully.")

    config = AutoConfig.from_pretrained(args_quant.input_path)
    quant_model_weight_path = os.path.join(args_quant.input_path, "quant_model_weight.safetensors")

    with open(os.path.join(args_quant.input_path, "quant_model_description.json"), "r") as f:
        quant_model_description = json.load(f)
    
    with open(os.path.join(args_quant.output_path, "quant_model_description.json"), "w") as f:
        f.write(json.dumps(quant_model_description, indent=4))

    model_layer_names = list(quant_model_description.keys())
    float_weight_dict = dict()

    quant_weight_dict = dict()
    for quant_param in ("weight", "weight_scale", "weight_offset"):
        quant_weight_dict[quant_param] = dict()
    
    print("[~] Loading weights onto CPU...")
    with safe_open(quant_model_weight_path, framework="pt", device="cpu") as f:
        model_layer_names = f.keys()
        for i in tqdm.tqdm(range(len(model_layer_names))):
            layer_name = model_layer_names[i]
            is_float = quant_model_description[layer_name] == "FLOAT"
            if is_float:
                float_weight_dict[layer_name] = f.get_tensor(layer_name)
            else:
                quant_param = layer_name.split(".")[-1]
                quant_weight_dict[quant_param][layer_name] = f.get_tensor(layer_name)
    print("[+] Weights loaded successfully.")

    cut_weights_config = CutWeightsConfig(float_weight_dict, args_quant.world_size, config)
    float_weight_parts = cut_weights(cut_weights_config)

    quant_weight_parts = dict()
    for quant_param in ("weight", "weight_scale", "weight_offset"):
        quant_weight_parts[quant_param] = cut_weights_quant(quant_weight_dict[quant_param],
                                                            args_quant.world_size,
                                                            config)

    save_dir_base = os.path.join(args_quant.output_path, 'part_model')
    print(f"[~] Saving the weights of the {args_quant.world_size} parts...")
    for i in tqdm.tqdm(range(args_quant.world_size)):
        save_dir = os.path.join(save_dir_base, str(i))
        os.makedirs(save_dir, exist_ok=True)
        config.to_json_file(os.path.join(save_dir, "config.json"))

        part_tensors = dict()
        for layer_name in model_layer_names:
            if "lm_head" in layer_name:
                continue
            is_float = quant_model_description[layer_name] == "FLOAT"
            if is_float:
                part_tensors[layer_name] = float_weight_parts[i][layer_name]
            else:
                quant_param = layer_name.split(".")[-1]
                part_tensors[layer_name] = quant_weight_parts[quant_param][i][layer_name]
            print(f"{layer_name:90} -- {part_tensors[layer_name].shape}")
        save_file(part_tensors, os.path.join(save_dir, "quant_model_weight.safetensors"))

    print("Tensor parallelism weights have been successfully saved.")
    print(f"the location of parallel quant weight is {args_quant.output_path}")


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
