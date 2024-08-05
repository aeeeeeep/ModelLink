# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import types
import logging as logger
import torch
from models import get_megatron_model

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')
    parser.add_argument('--add-qkv-bias', action='store_true',
                       help='Add bias for attention qkv', default=False,
    )
    parser.add_argument('--add-dense-bias', action='store_true',
                       help='Add bias for attention dense', default=False,
    )
    parser.add_argument('--embed-layernorm', action='store_true',
                       help='Add embed layernorm for word embedding', default=False,
    )
    parser.add_argument('--params-dtype', type=str,
                       help='Set weight dtype', default='fp16',
    )
    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Lora target modules.')
    group.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    group.add_argument('--lora-r', type=int, default=16,
                       help='Lora r.')
    group.add_argument('--lora-alpha', type=int, default=32,
                       help='Lora alpha.')


def build_metadata(args, margs):
    # Metadata.

    # Layernorm has bias; RMSNorm does not.
    if hasattr(margs, 'normalization'):
        norm_has_bias = margs.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = None
    md.checkpoint_args = margs
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.embed_layernorm = margs.embed_layernorm

    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    return md


def get_message_preprocess(models, args):
    # Send embeddings.
    tp_size = args.tensor_model_parallel_size
    message = {
        "word embeddings": torch.cat(
            [models.get_embedding_word_embeddings_weight(tp_rank=tp_rank) for tp_rank in range(tp_size)], dim=0
    )
    }
    if args.position_embedding_type == 'learned_absolute':
        message[f"position embeddings"] = models.get_embedding_position_embeddings_weight()
    if args.embed_layernorm:
        message[f"word embeddings norm_w"] = models.get_embedding_word_embeddings_norm_weight()
        message[f"word embeddings norm_b"] = models.get_embedding_word_embeddings_norm_bias()

    return message


def get_message_layer_norm(message, models, md, **kwargs):
    # Get non-parallel tensors from tp_rank 0.
    message["input norm weight"] = models.get_layers_input_layernorm_weight(**kwargs)
    if md.norm_has_bias:
        message["input norm bias"] = models.get_layers_input_layernorm_bias(**kwargs)

    message["post norm weight"] = models.get_layers_self_attention_post_attention_layernorm_weight(**kwargs)
    if md.norm_has_bias:
        message["post norm bias"] = models.get_layers_self_attention_post_attention_layernorm_bias(**kwargs)

    return message


def get_message_layer_attn(message, model, md=None, **kwargs):
    # Grab all parallel tensors for this layer
    qkv_weight = []
    qkv_bias = []
    dense_weight = []
    margs = model.get_args()

    for tp_rank in range(margs.tensor_model_parallel_size):
        kwargs["tp_rank"] = tp_rank
        qkv_weight.append(model.get_layers_self_attention_linear_qkv_weight(**kwargs))
        dense_weight.append(model.get_layers_self_attention_linear_proj_weight(**kwargs))

        if md.linear_bias or margs.add_qkv_bias:
            qkv_bias.append(model.get_layers_self_attention_linear_qkv_bias(**kwargs))

    # Handle gated linear units
    # simple concat of the rest
    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
    message["dense weight"] = torch.cat(dense_weight, dim=1)
    if md.linear_bias or margs.add_qkv_bias:
        message["qkv bias"] = torch.cat(qkv_bias, dim=0)
    if margs.add_qkv_bias:
        message["qkv bias"] = torch.cat(qkv_bias, dim=0)

    if md.linear_bias:
        message["dense bias"] = model.get_layers_self_attention_linear_proj_bias(**kwargs)

    return message


def _get_message_layer_mlp(message, model, md=None, **kwargs):
    margs = model.get_args()
    mlp_l0_weight = []
    mlp_l1_weight = []
    mlp_l0_bias = []
    for tp_rank in range(margs.tensor_model_parallel_size):
        kwargs["tp_rank"] = tp_rank
        mlp_l0_weight.append(model.get_layers_mlp_linear_fc1_weight(**kwargs))
        mlp_l1_weight.append(model.get_layers_mlp_linear_fc2_weight(**kwargs))
        if md.linear_bias:
            mlp_l0_bias.append(model.get_layers_mlp_linear_fc1_bias(**kwargs))

    # Handle gated linear units
    if md.swiglu:
        # concat all the first halves ('W's) and all the second halves ('V's)
        for tp_rank in range(margs.tensor_model_parallel_size):
            mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
        message[f"mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
        message[f"mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
    else:
        message[f"mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

    # simple concat of the rest
    message[f"mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
    if md.linear_bias:
        if md.swiglu:
            for tp_rank in range(margs.tensor_model_parallel_size):
                mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
            message[f"mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
            message[f"mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
        else:
            message[f"mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)


def get_message_layer_mlp(message, model, md=None, **kwargs):
    # Grab all parallel tensors for this layer
    margs = model.get_args()
    if margs.num_experts:
        message["mlp_moe"] = {}
        num_experts_local = margs.num_experts // margs.expert_model_parallel_size
        mlp_router_weight = model.get_layers_mlp_router_weight(**kwargs)
        message["mlp_moe"]["mlp router weight"] = mlp_router_weight
        for ep_rank in range(margs.expert_model_parallel_size):
            kwargs["ep_rank"] = ep_rank
            for expert_idx in range(num_experts_local):
                kwargs["expert_idx"] = expert_idx
                global_expert_idx = expert_idx + ep_rank * num_experts_local
                message["mlp_moe"][f"expert {global_expert_idx}"] = {}
                expert = message["mlp_moe"][f"expert {global_expert_idx}"]
                _get_message_layer_mlp(expert, model, md, **kwargs)
    else:
        _get_message_layer_mlp(message, model, md, **kwargs)

    return message


def get_message_postprocess(model, md, **kwargs):
    # Send final norm from tp_rank 0.
    message = {}
    message[f"weight"] = model.get_final_layernorm_weight(**kwargs)
    if md.norm_has_bias:
        message[f"bias"] = model.get_final_layernorm_bias(**kwargs)

    return message


def get_message_output_layer(model, md, **kwargs):
    # Send final norm from tp_rank 0.
    margs = model.get_args()
    tp_size = margs.tensor_model_parallel_size
    message = {}
    if md.output_layer:
        get_output_layer_weight_list = []
        for tp_rank in range(tp_size):
            kwargs["tp_rank"] = tp_rank
            get_output_layer_weight_list.append(
                model.get_output_layer_weight(**kwargs)
            )
        message[f"weight"] = torch.cat(get_output_layer_weight_list, dim=0)

    return message


def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    model_mg = get_megatron_model(args_cmd=args)
    model_mg.initialize_megatron_args(queue=queue, loader_megatron=True)

    model_mg.set_tensor_model_parallel_world_size(model_mg.args.tensor_model_parallel_size)
    model_mg.set_expert_model_parallel_world_size(model_mg.args.expert_model_parallel_size)
    model_mg.set_pipeline_model_parallel_world_size(model_mg.args.pipeline_model_parallel_size)
    model_mg.set_virtual_pipeline_model_parallel_world_size(model_mg.args.virtual_pipeline_model_parallel_size)

    # Get first pipe stage.
    model_mg.set_tensor_model_parallel_rank(0)
    model_mg.set_pipeline_model_parallel_rank(0)

    margs = model_mg.get_args()

    md = build_metadata(args, margs)
    queue.put(md)
    model_mg.get_modules_from_pretrained(pp_stage_cache_flag=True)

    def queue_put(name, msg):
        logger.info(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = get_message_preprocess(model_mg, margs)
    queue_put("embeddings", message)

    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    total_layer_num = 0
    for vp_rank in range(vp_size):
        for pp_rank in range(pp_size):
            model_mg.set_pipeline_model_parallel_rank(pp_rank)
            model_mg.get_modules_from_pretrained(pp_stage_cache_flag=True)
            kwargs = {"vp_rank": vp_rank}
            for layer_num in range(len(model_mg.get_layers_module(**kwargs))):
                kwargs["layer_num"] = layer_num
                message = {}
                message = get_message_layer_norm(message, model_mg, md, **kwargs)
                message = get_message_layer_attn(message, model_mg, md, **kwargs)
                message = get_message_layer_mlp(message, model_mg, md, **kwargs)
                queue_put(f"transformer layer {total_layer_num}", message)
                total_layer_num = total_layer_num + 1

    kwargs = {"pp_rank": pp_size - 1, "vp_rank": vp_size - 1}
    message = get_message_postprocess(model_mg, md, **kwargs)
    queue_put("final norm", message)

    message = get_message_output_layer(model_mg, md, **kwargs)
    if message:
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
