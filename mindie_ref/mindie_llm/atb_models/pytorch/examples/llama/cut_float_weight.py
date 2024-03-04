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
import os
import argparse
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig


# cut float weights
# cut_row_keys :dim 0  cut_col_keys :dim 1  nn.linear: x*A.T
def cut_weights(model_in, world_size, cut_row_keys=('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'),
                cut_col_keys=('o_proj', 'down_proj'), is_yi6b=0):
    new_state_dict_list = [{} for i in range(world_size)]
    for key, tensor in model_in.state_dict().items():
        key_short = key.split('.')[-2]
        if key_short in cut_row_keys:
            if is_yi6b and key_short in ['k_proj', 'v_proj']:
                # 适配num_key_value_heads=4, world_size=8
                cut_tensor_list = torch.chunk(tensor, min(4, world_size), dim=0)
            else:
                cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        else:
            cut_tensor_list = [tensor] * world_size
        for tmp_world in range(world_size):
            if is_yi6b and key_short in ['k_proj', 'v_proj']:
                # 适配num_key_value_heads=4, world_size=8
                new_state_dict_list[tmp_world][key] = cut_tensor_list[tmp_world // 2]
            else:
                new_state_dict_list[tmp_world][key] = cut_tensor_list[tmp_world]
    return new_state_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="/data/models/llama2-7b",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/data/models/llama2-7b-part_model_2',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
    )
    parser.add_argument(
        "--cut_row_keys",
        default=('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'),
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default=('o_proj', 'down_proj'),
        help="cut_col_keys",
    )
    parser.add_argument(
        "--is_yi6b",
        type=int,
        default=0,
        help="set specified case_pair if 1",
    )
    args = parser.parse_args()
    args.world_size = int(args.world_size) 
    tokenizer_config_path = os.path.join(args.input_path, 'tokenizer_config.json')
    with open(tokenizer_config_path) as f:
        tokenizer_config = json.load(f)
        use_fast = tokenizer_config.get('use_fast', True)
    tokenizer = LlamaTokenizer.from_pretrained(args.input_path, use_fast=use_fast)
    tokenizer.save_pretrained(os.path.join(args.output_path, 'tokenizer'))
    
    model = LlamaForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.float16)
    state_dict_list = cut_weights(model, args.world_size, args.cut_row_keys, args.cut_col_keys, args.is_yi6b)
    model_config = model.config
    # create new model config, add the world size parameter
    # the model size will be cut according to the world size in the model file
    if hasattr(model_config, 'num_key_value_heads'):
        num_key_value_heads = model_config.num_key_value_heads
    else:
        num_key_value_heads = model_config.num_attention_heads
    if hasattr(model_config, 'pretraining_tp'):
        pretraining_tp = model_config.pretraining_tp
    else:
        pretraining_tp = 1
    if hasattr(model_config, 'rope_scaling'):
        rope_scaling = model_config.rope_scaling
    else:
        rope_scaling = None
    if hasattr(model_config, 'rope_theta'):
        rope_theta = model_config.rope_theta
    else:
        rope_theta = 10000
    if hasattr(model_config, 'attention_bias'):
        attention_bias = model_config.attention_bias
    else:
        attention_bias = False
    create_config = LlamaConfig(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=num_key_value_heads * 2 if args.is_yi6b else num_key_value_heads,
            hidden_act=model_config.hidden_act,
            max_position_embeddings=model_config.max_position_embeddings,
            initializer_range=model_config.initializer_range,
            rms_norm_eps=model_config.rms_norm_eps,
            use_cache=model_config.use_cache,
            pad_token_id=model_config.pad_token_id,
            bos_token_id=model_config.bos_token_id,
            eos_token_id=model_config.eos_token_id,
            pretraining_tp=pretraining_tp,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            world_size=args.world_size,
            architectures=model_config.architectures,
            model_type=model_config.model_type,
            torch_dtype=model_config.torch_dtype,
            transformers_version=model_config.transformers_version
    )
    # create new model according to the model config
    creat_model = LlamaForCausalLM(create_config)
    for i in range(args.world_size):
        creat_model.load_state_dict(state_dict_list[i]) # load the weights to the model
        creat_model.save_pretrained(os.path.join(args.output_path, 'part_model', str(i))) # save model
    print('Tensor parallelism weights have been successfully saved.')
