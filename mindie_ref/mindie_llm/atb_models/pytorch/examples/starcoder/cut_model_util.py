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
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pdb


#cut weights
#cut_row_keys :dim 0  cut_col_keys :dim 1  nn.linear: x*A.T
def cut_weights(model, world_size, cut_row_keys=['c_fc'],
                cut_col_keys=['c_proj'], special_cut=['c_attn']):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in model.state_dict().items():
        key_short = key.split('.')[-2]  # 包含weight与bias
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            # 横切只切weight，
            if key.split('.')[-1] == "weight":
                cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
            else:
                # 不切bias，适配加速库的all reduce, 先matmul-> all_reduce->add bias
                cut_tensor_list = [tensor] * world_size
        elif key_short in special_cut:
            embed_dim = 6144
            kv_dim = 128
            q, kv = tensor.split((embed_dim, 2 * kv_dim), dim=0)
            print(q.shape, kv.shape)
            chunk_tensors = torch.chunk(q, world_size, dim=0)
            cut_tensor_list=[]
            for i in range(len(chunk_tensors)):
                cut_tensor_list.append(torch.cat((chunk_tensors[i], kv), 0))
                print(cut_tensor_list[i].shape)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="/data/models/starcoder",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/data/models/starcoder-part_model',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
    )
    args = parser.parse_args()
    args.world_size = int(args.world_size) 
    tokenizer = AutoTokenizer.from_pretrained(args.input_path, use_fast=False)
    tokenizer.save_pretrained(os.path.join(args.output_path, 'tokenizer'))
    
    model = AutoModelForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.float16)
    state_dict_list = cut_weights(model, args.world_size)
    model_config = model.config
    new_config = model_config
    new_config.world_size = args.world_size
    # create new model according to the model config
    new_model = AutoModelForCausalLM.from_config(new_config)
    for i in range(args.world_size):
        new_model.load_state_dict(state_dict_list[i]) # load the weights to the model
        new_model.save_pretrained(os.path.join(args.output_path, 'part_model', str(i))) # save model
    print('Tensor parallelism weights have been successfully saved.')


