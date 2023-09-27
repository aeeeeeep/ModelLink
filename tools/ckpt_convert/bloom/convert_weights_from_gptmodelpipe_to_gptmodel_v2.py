import argparse
import json
import math
import os
import sys
import torch

work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
print(f"work_path : {os.path.abspath(work_path)}")
sys.path.append(work_path)

from ckpt_utils import make_ascendspeed_model_dirs


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-model-dir", type=str, default="./input_model_dir", help="bloom huggingface model dir")
    parser.add_argument("--output-model-dir", type=str, default="./output_model_dir",
                        help="bloom ascendspeed model dir")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1, help="degree of tensor model parallel")
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1,
                        help="degree of pipeline model parallel")
    parser.add_argument("--type", type=str, choices=["7B", "176B"], default="7B")
    return parser.parse_args()


model_config = {
    "7B": [30, 4096, 32],  # num_layers, hidden_size, num_attention_heads
    "176B": [70, 14336, 112]
}


def extract_gptmodelpipe(input_model_dir):
    files = os.listdir(os.path.join(input_model_dir, f'global_step1000'))
    input_models = {}
    for f in files:
        ckpt_file_path = os.path.join(input_model_dir, f'global_step1000', f)
        input_models[f] = torch.load(ckpt_file_path, map_location="cpu")
    print(f"load gpt model pipe finish")
    return input_models


def generate_gptmodel_weights(input_model_dir, output_model_dir, tensor_model_parallel_size,
                              pipeline_model_parallel_size, type):
    layer_size = model_config.get(type)[0]
    gptmodelpipe = extract_gptmodelpipe(input_model_dir)

    ### 实际上有的参数文件集合
    param_file_set = [os.path.basename(file_path) for file_path in gptmodelpipe.keys()]
    param_file_set = sorted([file_name for file_name in param_file_set if file_name.startswith('layer_')])
    print(f"文件集合 : {param_file_set}")

    release_model_dir = os.path.join(output_model_dir, "release")
    language_model = {}
    language_model['encoder'] = {}

    word_embeddings_for_head = {}

    for pp_rank in range(pipeline_model_parallel_size):
        layer_mean = math.ceil(layer_size / pipeline_model_parallel_size)
        current_layer_pp_rank = list(range(pp_rank * layer_mean + 3, (pp_rank + 1) * layer_mean + 3))
        if pp_rank == 0:
            current_layer_pp_rank.append(1)
        if pp_rank == pipeline_model_parallel_size - 1:
            current_layer_pp_rank = list(range(pp_rank * layer_mean + 3, layer_size + 3))
            current_layer_pp_rank.append(1)
            current_layer_pp_rank.append(layer_size + 4)

        ### 原理上应该有的参数文件集合
        theo_file_set = [f"layer_{layer_num:02d}-model_{tp_rank:02d}-model_states.pt" for layer_num in
                         current_layer_pp_rank for tp_rank in range(tensor_model_parallel_size)]
        print(f"原理文件集合 : {theo_file_set}")

        if len(set(param_file_set) & set(theo_file_set)) == len(set(param_file_set)):
            print(f"current rank : {pp_rank}, 包含的层 ：{current_layer_pp_rank}")
        else:
            print(f"{current_layer_pp_rank} 不在rank: {pp_rank}")
            continue

        for tp_rank in range(tensor_model_parallel_size):
            for layer_num in current_layer_pp_rank:
                layer_name = f"layer_{layer_num:02d}-model_{tp_rank:02d}-model_states.pt"
                if layer_num == 1:
                    if pp_rank == 0:
                        language_model['embedding'] = {}
                        language_model['embedding']['word_embeddings'] = {}
                        language_model['embedding']['word_embeddings']['weight'] = gptmodelpipe[layer_name][
                            'word_embeddings.weight']

                    if pp_rank == pipeline_model_parallel_size - 1:
                        word_embeddings_for_head['weight'] = gptmodelpipe[layer_name][
                            'word_embeddings.weight']

                elif layer_num == 2:
                    continue
                elif layer_num == layer_size + 3:
                    continue
                elif layer_num == layer_size + 4:
                    language_model['encoder']["final_layernorm.weight"] = gptmodelpipe[layer_name][
                        "final_layernorm.weight"]
                    language_model['encoder']["final_layernorm.bias"] = gptmodelpipe[layer_name]["final_layernorm.bias"]
                else:
                    # encoder_layer_name = f"layers.{layer_num - 3}."
                    encoder_layer_name = f"layers.{layer_num - 3 - pp_rank * layer_mean}."

                    language_model['encoder'][f"{encoder_layer_name}input_layernorm.weight"] = gptmodelpipe[layer_name][
                        'input_layernorm.weight']
                    language_model['encoder'][f"{encoder_layer_name}input_layernorm.bias"] = gptmodelpipe[layer_name][
                        'input_layernorm.bias']
                    language_model['encoder'][f"{encoder_layer_name}self_attention.query_key_value.weight"] = \
                        gptmodelpipe[layer_name]['self_attention.query_key_value.weight']
                    language_model['encoder'][f"{encoder_layer_name}self_attention.query_key_value.bias"] = \
                        gptmodelpipe[layer_name]['self_attention.query_key_value.bias']
                    language_model['encoder'][f"{encoder_layer_name}self_attention.dense.weight"] = \
                        gptmodelpipe[layer_name]['self_attention.dense.weight']
                    language_model['encoder'][f"{encoder_layer_name}self_attention.dense.bias"] = \
                        gptmodelpipe[layer_name][
                            'self_attention.dense.bias']
                    language_model['encoder'][f"{encoder_layer_name}post_attention_layernorm.weight"] = \
                        gptmodelpipe[layer_name]['post_attention_layernorm.weight']
                    language_model['encoder'][f"{encoder_layer_name}post_attention_layernorm.bias"] = \
                        gptmodelpipe[layer_name]['post_attention_layernorm.bias']
                    language_model['encoder'][f"{encoder_layer_name}mlp.dense_h_to_4h.weight"] = \
                        gptmodelpipe[layer_name][
                            'mlp.dense_h_to_4h.weight']
                    language_model['encoder'][f"{encoder_layer_name}mlp.dense_h_to_4h.bias"] = gptmodelpipe[layer_name][
                        'mlp.dense_h_to_4h.bias']
                    language_model['encoder'][f"{encoder_layer_name}mlp.dense_4h_to_h.weight"] = \
                        gptmodelpipe[layer_name][
                            'mlp.dense_4h_to_h.weight']
                    language_model['encoder'][f"{encoder_layer_name}mlp.dense_4h_to_h.bias"] = gptmodelpipe[layer_name][
                        'mlp.dense_4h_to_h.bias']
            model_dic = {'checkpoint_version': 3.0, 'model': {}}
            model_dic['model']['language_model'] = language_model
            if pp_rank == pipeline_model_parallel_size - 1:
                model_dic['model']['word_embeddings_for_head'] = word_embeddings_for_head

            print(model_dic['model'].keys())
            model_dir = os.path.join(release_model_dir, f"{'mp_rank_{:02d}_{:03d}'.format(tp_rank, pp_rank)}")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model_dic, os.path.join(model_dir, "model_optim_rng.pt"))
            print(f'{os.path.join(model_dir, "model_optim_rng.pt")} saved!')
        # 找到对应的rank，保存结束
        break
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>change over>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


if __name__ == '__main__':
    args = get_args()
    make_ascendspeed_model_dirs(args.output_model_dir)
    generate_gptmodel_weights(args.input_model_dir, args.output_model_dir, args.tensor_model_parallel_size,
                              args.pipeline_model_parallel_size, args.type)
