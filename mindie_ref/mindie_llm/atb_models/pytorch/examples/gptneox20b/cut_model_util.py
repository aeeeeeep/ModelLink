#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
"""
cut model
@create: 2024/1/26 14:53 
@since: 2024/1/26 14:53 
"""

import os
import argparse
import shutil

import torch

from transformers import AutoModelForCausalLM


def cut_weights(model, world_size, cut_row_keys=['query_key_value', 'dense_h_to_4h','embed_out'],
                cut_col_keys=['dense', 'dense_4h_to_h']):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in model.state_dict().items():
        key_short = key.split('.')[-2]
        cut_tensor_list_t = []
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            if key.split('.')[-1] == "weight":
                cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
            else:
                cut_tensor_list = [tensor.div(world_size)] * world_size

        else:
            cut_tensor_list = [tensor] * world_size
        for cut_tensor in cut_tensor_list:
            cut_tensor_list_t.append(cut_tensor.clone())
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list_t[i]
    return state_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="/home/lfy/LM_trans/gptneox20b/model",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/home/lfy/LM_trans/gptneox20b/model/part_model',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
    )
    parser.add_argument(
        "--cut_row_keys",
        default=['query_key_value', 'dense_h_to_4h', 'embed_out'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default=['dense', 'dense_4h_to_h'],
        help="cut_col_keys",
    )

    args = parser.parse_args()

    args.world_size = int(args.world_size)

    model_path = args.input_path
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()  # 都加载模型和权重
    state_dict_list = cut_weights(model, args.world_size, args.cut_row_keys, args.cut_col_keys)
    model_config = model.config
    model_config.world_size = args.world_size

    creat_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    for i in range(args.world_size):

        creat_model.load_state_dict(state_dict_list[i])

        target_dir = os.path.join(args.output_path, str(i))
        os.makedirs(target_dir, exist_ok=True)
        creat_model.save_pretrained(target_dir)
        creat_model.config.auto_map["AutoModelForCausalLM"] = "modeling_gpt_neox_ascend.GPTNeoXForCausalLM"
        creat_model.config.save_pretrained(target_dir)
        for source_file in ["configuration_gpt_neox.py", "merges.txt", "special_tokens_map.json", "tokenizer.json",
                            "tokenizer_config.json",
                            "vocab.json",
                            "modeling_gpt_neox_ascend.py"]:
            shutil.copy(os.path.join(model_path, source_file), target_dir)

    print('Tensor parallelism weights have been successfully saved.')
