# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import argparse
import os.path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Use the Sparse Upcycling method to convert hf dense model parameters into moe model parameters.",
        allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--hf-dense-model-files', required=True, nargs='+',
                        help='Parameter file of the huggingface dense model, for example, pytorch_model.bin')
    parser.add_argument('--hf-moe-output_dir', type=str, required=True,
                        help='Directory for storing MOE model output files.')
    parser.add_argument('--num-experts', type=int, required=True,
                        help='Number of MOE experts')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of transformer layers.')
    parser.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                        help='Untie embeddings and output weights.'),
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed used for python, pytorch.')
    known_args, _ = parser.parse_known_args()

    hf_dense_model_files = known_args.hf_dense_model_files
    hf_moe_output_dir = known_args.hf_moe_output_dir
    num_experts = known_args.num_experts
    num_layers = known_args.num_layers
    untie_embeddings_and_output_weights = known_args.untie_embeddings_and_output_weights
    torch.manual_seed(seed=known_args.seed)

    print(f'model files : {hf_dense_model_files}')
    model_dir = os.path.dirname(hf_dense_model_files[0])
    for i, model_path in enumerate(hf_dense_model_files):
        model_path = model_path if i == 0 else os.path.join(model_dir, model_path)
        if not os.path.exists(model_path):
            raise ValueError(f"model path is not exists : {model_path}")

        model = torch.load(model_path, map_location='cpu')
        moe_model = {"model.embed_tokens.weight": model["model.embed_tokens.weight"],
                     "model.norm.weight": model["model.norm.weight"]}
        if untie_embeddings_and_output_weights:
            moe_model['output_layer.weight'] = model['output_layer.weight']

        for layer_idx in range(num_layers):
            moe_model[f'model.layers.{layer_idx}.input_layernorm.weight'] = model[
                f'model.layers.{layer_idx}.input_layernorm.weight']
            moe_model[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = model[
                f'model.layers.{layer_idx}.self_attn.q_proj.weight']
            moe_model[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = model[
                f'model.layers.{layer_idx}.self_attn.k_proj.weight']
            moe_model[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = model[
                f'model.layers.{layer_idx}.self_attn.v_proj.weight']
            moe_model[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = model[
                f'model.layers.{layer_idx}.self_attn.o_proj.weight']
            moe_model[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = model[
                f'model.layers.{layer_idx}.post_attention_layernorm.weight']

            hidden_size = model[f'model.layers.{layer_idx}.input_layernorm.weight'].shape[0]
            moe_model[f'model.layers.{layer_idx}.mlp.gate.weight'] = torch.normal(mean=0, std=1e-2,
                                                                                  size=(num_experts, hidden_size))
            for expert_i in range(num_experts):
                moe_model[f'model.layers.{layer_idx}.mlp.experts.{expert_i}.w1.weight'] = model[
                    f'model.layers.{layer_idx}.mlp.gate_proj.weight']
                moe_model[f'model.layers.{layer_idx}.mlp.experts.{expert_i}.w2.weight'] = model[
                    f'model.layers.{layer_idx}.mlp.down_proj.weight']
                moe_model[f'model.layers.{layer_idx}.mlp.experts.{expert_i}.w3.weight'] = model[
                    f'model.layers.{layer_idx}.mlp.up_proj.weight']

        if not os.path.exists(hf_moe_output_dir):
            os.makedirs(hf_moe_output_dir, exist_ok=True)
        filename = os.path.basename(model_path)
        save_path = os.path.join(hf_moe_output_dir, filename)
        torch.save(moe_model, save_path)


if __name__ == '__main__':
    main()
