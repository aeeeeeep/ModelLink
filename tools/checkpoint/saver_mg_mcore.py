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

import os
import sys
import time
import copy
import torch
from megatron.training.checkpointing import save_checkpoint
from megatron.core import mpu
from models import get_megatron_model


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                            'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                            'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--save-model-type', type=str, default='megatron',
                       choices=['megatron'], help='Save model type')
    group.add_argument("--w-pack", type=bool,
                       help='True is w_pack weight for llm',
                       default=False)
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--num-layer-list',
                       type=str, help='a list of number of layers, seperated by comma; e.g., 4,4,4,4')


def update_padded_vocab_size(md, model_mg, orig_tensor, orig_word_embed):
    # figure out what our padded vocab size is
    if md.true_vocab_size is not None:
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        margs = model_mg.get_args()
        padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)
        model_mg.set_padded_vocab_size(padded_vocab_size)
    else:
        print("Original vocab size not specified, leaving embedding table as-is. "
              "If you've changed the tensor parallel size this could cause problems.")
        model_mg.set_padded_vocab_size(orig_word_embed.shape[0])
    margs = model_mg.get_args()
    padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)
    model_mg.set_padded_vocab_size(padded_vocab_size)


def vocab_padding(orig_vocab_size, padded_vocab_size, orig_tensor):
    # figure out what our padded vocab size is

    # Cut out extra padding we don't need
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = orig_tensor[0:padded_vocab_size, :]

    # Expanding embedding to larger size by replicating final entry
    elif orig_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - orig_vocab_size

        full_word_embed = torch.cat((
            orig_tensor,
            orig_tensor[-1].unsqueeze(0).expand(padding_size, -1)))

    # Same size!
    else:
        full_word_embed = orig_tensor

    return full_word_embed


def reset_cmd_args_from_md(args, md):
    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print(
                "loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                "Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print(
                "loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                "Default to 1.")
            args.target_pipeline_parallel_size = 1


def set_model_preprocess(model, embeddings_msg, check_message):
    md = model.get_metadata()
    margs = model.get_args()
    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    orig_word_embed_n_w, orig_word_embed_n_b = None, None
    if "word embeddings norm_w" in embeddings_msg and "word embeddings norm_b" in embeddings_msg:
        orig_word_embed_n_w = embeddings_msg.pop("word embeddings norm_w")
        orig_word_embed_n_b = embeddings_msg.pop("word embeddings norm_b")
    check_message(embeddings_msg)

    if md.true_vocab_size is not None:
        orig_vocab_size = orig_word_embed.shape[0]
        full_word_embed = vocab_padding(orig_vocab_size, margs.padded_vocab_size, orig_word_embed)
    else:
        full_word_embed = orig_word_embed

    # Split into new tensor model parallel sizes  tensor_model_parallel_size
    out_word_embed = torch.chunk(full_word_embed, margs.tensor_model_parallel_size, dim=0)

    modules_count = model.get_modules_count()
    for tp_rank in range(modules_count):
        model.set_embedding_word_embeddings_weight(tp_rank=tp_rank, data=out_word_embed[tp_rank])
        if orig_word_embed_n_w is not None:
            model.set_embedding_word_embeddings_norm_weight(tp_rank=tp_rank, data=orig_word_embed_n_w)
            model.set_embedding_word_embeddings_norm_bias(tp_rank=tp_rank, data=orig_word_embed_n_b)
        if pos_embed is not None:
            model.set_embedding_position_embeddings_weight(tp_rank=tp_rank, data=pos_embed)
        else:
            if hasattr(model.get_embedding_module(), 'position_embeddings'):
                raise ValueError("model should have position_embeddings")

    return out_word_embed


def save_model_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    md = queue_get()
    reset_cmd_args_from_md(args, md)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    model_mg = get_megatron_model(args_cmd=args, md=md)
    model_mg.initialize_megatron_args(queue=queue)

    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    model_mg.get_modules_from_config(args.target_tensor_parallel_size, pre_process=True, post_process=post_process)

    # Embeddings
    embeddings_msg = queue_get("embeddings")
    out_word_embed = set_model_preprocess(model_mg, embeddings_msg, check_message)

    margs = model_mg.get_args()

    # Transformer layers
    # -------------------
    total_layer_num = 0
    lst = []
    if args.num_layers_per_virtual_pipeline_stage and args.save_model_type == 'megatron':
        times = 3
        while queue.qsize() > 3 or times >= 0:
            if times >= 0:
                time.sleep(1)
                times -= 1
                continue
            lst.append(queue.get())
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
            model_mg.get_modules_from_config(args.target_tensor_parallel_size, False, post_process)

        if args.num_layers_per_virtual_pipeline_stage and args.save_model_type == 'megatron':
            vp_size = margs.num_layers // args.target_pipeline_parallel_size // args.num_layers_per_virtual_pipeline_stage
        else:
            vp_size = 1
        for vpp_rank in range(vp_size):
            for layer in range(len(model_mg.get_layers_module()) // vp_size):
                if args.num_layers_per_virtual_pipeline_stage and args.save_model_type == 'megatron':
                    # The execution order between layers in the VPP model is different from that in the PP model. Here,
                    # it is necessary to calculate the index and arrange the layers in the actual execution order
                    total_layer_num = args.target_pipeline_parallel_size * vpp_rank * args.num_layers_per_virtual_pipeline_stage + pp_rank * args.num_layers_per_virtual_pipeline_stage + layer
                    msg = lst[total_layer_num]
                else:
                    msg = queue_get(f"transformer layer {total_layer_num}")

                # duplicated tensors
                input_norm_weight = msg.pop("input norm weight")
                if md.norm_has_bias:
                    input_norm_bias = msg.pop("input norm bias")
                post_norm_weight = msg.pop("post norm weight")
                if md.norm_has_bias:
                    post_norm_bias = msg.pop("post norm bias")
                if md.linear_bias:
                    dense_bias = msg.pop("dense bias")
                    mlp_l1_bias = msg.pop("mlp l1 bias")

                if args.add_qkv_bias:
                    qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                if args.add_dense_bias:
                    dense_bias = msg.pop("dense bias")

                qkv_org = msg.pop("qkv weight")
                qkv_weight = torch.chunk(qkv_org, args.target_tensor_parallel_size, dim=0)

                # Split up the parallel tensors
                dense_weight = torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1)
                mlp_l1_weight = torch.chunk(msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1)

                # Special handling for swiglu
                if md.swiglu:
                    mlp_l0_weight_W = torch.chunk(msg.pop("mlp l0 weight W"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_weight_V = torch.chunk(msg.pop("mlp l0 weight V"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
                else:
                    mlp_l0_weight = torch.chunk(msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0)

                if md.linear_bias:
                    qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                    if md.swiglu:
                        mlp_l0_bias_W = torch.chunk(msg.pop("mlp l0 bias W"), args.target_tensor_parallel_size, dim=0)
                        mlp_l0_bias_V = torch.chunk(msg.pop("mlp l0 bias V"), args.target_tensor_parallel_size, dim=0)
                        mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
                    else:
                        mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0)

                # Save them to the model
                for tp_rank in range(args.target_tensor_parallel_size):
                    if args.num_layers_per_virtual_pipeline_stage and args.save_model_type == 'megatron':
                        l_idx = vpp_rank * args.num_layers_per_virtual_pipeline_stage + layer
                    else:
                        l_idx = layer

                    model_mg.set_layers_input_layernorm_weight(tp_rank=tp_rank, layer_idx=l_idx, data=input_norm_weight)
                    if md.norm_has_bias:
                        model_mg.set_layers_input_layernorm_bias(tp_rank=tp_rank, layer_idx=l_idx, data=input_norm_bias)
                    model_mg.set_layers_self_attention_linear_qkv_weight(tp_rank=tp_rank, layer_idx=l_idx, data=qkv_weight[tp_rank])
                    model_mg.set_layers_self_attention_linear_proj_weight(tp_rank=tp_rank, layer_idx=l_idx, data=dense_weight[tp_rank])
                    model_mg.set_layers_self_attention_pre_mlp_layernorm_weight(tp_rank=tp_rank, layer_idx=l_idx, data=post_norm_weight)
                    if md.norm_has_bias:
                        model_mg.set_layers_self_attention_pre_mlp_layernorm_bias(tp_rank=tp_rank, layer_idx=l_idx, data=post_norm_bias)

                    model_mg.set_layers_mlp_linear_fc1_weight(tp_rank=tp_rank, layer_idx=l_idx, data=mlp_l0_weight[tp_rank])
                    model_mg.set_layers_mlp_linear_fc2_weight(tp_rank=tp_rank, layer_idx=l_idx, data=mlp_l1_weight[tp_rank])

                    if md.linear_bias:
                        model_mg.set_layers_self_attention_linear_qkv_bias(tp_rank=tp_rank, layer_idx=l_idx, data=qkv_bias[tp_rank])
                        model_mg.set_layers_self_attention_linear_proj_bias(tp_rank=tp_rank, layer_idx=l_idx, data=dense_bias)
                        model_mg.set_layers_mlp_linear_fc1_bias(tp_rank=tp_rank, layer_idx=l_idx, data=mlp_l0_bias[tp_rank])
                        model_mg.set_layers_mlp_linear_fc2_bias(tp_rank=tp_rank, layer_idx=l_idx, data=mlp_l1_bias)

                    if args.add_qkv_bias:
                        model_mg.set_layers_self_attention_linear_qkv_bias(tp_rank=tp_rank, layer_idx=l_idx, data=qkv_bias[tp_rank])
                    if args.add_dense_bias:
                        model_mg.set_layers_self_attention_linear_proj_bias(tp_rank=tp_rank, layer_idx=l_idx, data=dense_bias)

                total_layer_num = total_layer_num + 1
                check_message(msg)

        if post_process:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                model_mg.set_final_layernorm_weight(tp_rank=tp_rank, data=final_norm_weight)
                if md.norm_has_bias:
                    model_mg.set_final_layernorm_bias(tp_rank=tp_rank, data=final_norm_bias)
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    model_mg.set_output_layer_weight(tp_rank=tp_rank, data=out_word_embed[tp_rank])
            del final_norm_weight
            if md.norm_has_bias:
                del final_norm_bias
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                output_layer = msg.pop("weight")
                if md.true_vocab_size is not None:
                    orig_vocab_size = output_layer.shape[0]
                    full_word_embed = vocab_padding(orig_vocab_size, margs.padded_vocab_size, output_layer)
                else:
                    full_word_embed = output_layer

                output_layer_weight = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)

                for tp_rank in range(args.target_tensor_parallel_size):
                    model_mg.set_output_layer_weight(tp_rank=tp_rank, data=output_layer_weight[tp_rank])
                del output_layer_weight
                check_message(msg)
            msg = queue_get()

        for tp_rank in range(args.target_tensor_parallel_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)
            # Split the PP into multiple VPPs and select the corresponding layers for each VPP by copying and deleting
            if args.num_layers_per_virtual_pipeline_stage:
                vp_models = []
                layers = margs.num_layers // args.target_pipeline_parallel_size
                for vp_rank in range(vp_size):
                    model = copy.deepcopy(model_mg.get_model_module(tp_rank=tp_rank))
                    left = vp_rank * args.num_layers_per_virtual_pipeline_stage
                    right = (vp_rank + 1) * args.num_layers_per_virtual_pipeline_stage
                    for i in range(layers - 1, -1, -1):
                        if i >= right or i < left:
                            del model.decoder.layers[i]
                    if right < layers and pp_rank == args.target_pipeline_parallel_size - 1:
                        del model.decoder.final_layernorm
                        if getattr(model, "output_layer", None):
                            model.post_process = False
                            del model.output_layer
                    if pp_rank == 0 and vp_rank > 0:
                        model.pre_process = False
                        del model.embedding
                    vp_models.append(model)
                save_checkpoint(md.iteration, vp_models, None, None, 0)
            else:
                save_checkpoint(md.iteration, [model_mg.get_model_module(tp_rank=tp_rank)], None, None, 0)
    print("Done!")
