# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import types

import torch


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

def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from ascendspeed import megatron_adaptor
        from megatron.arguments import validate_args
        from modellink.utils import parse_args
        from megatron.global_vars import set_args, set_global_variables
        from megatron.checkpointing import load_args_from_checkpoint
        from megatron.checkpointing import load_checkpoint as load_checkpoint_mg
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.embed_layernorm = args.embed_layernorm
    margs, checkpoint_args = load_args_from_checkpoint(margs)
    margs.add_qkv_bias = args.add_qkv_bias
    margs.add_dense_bias = args.add_dense_bias
    if args.add_dense_bias:
        margs.skip_bias_add = False
    if args.params_dtype == 'bf16':
        margs.bf16 = True
    elif args.params_dtype == 'fp16':
        margs.fp16 = True

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    consumed_train_samples = None
    consumed_valid_samples = None

    def get_models(count, dtype):
        nonlocal consumed_train_samples
        nonlocal consumed_valid_samples
        model_array_len = margs.virtual_pipeline_model_parallel_size
        if model_array_len is None:
            model_array_len = 1
        models = [[] for _ in range(model_array_len)]
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        for rank in range(count):
            mpu.set_tensor_model_parallel_rank(rank)
            if margs.virtual_pipeline_model_parallel_size is not None:
                model_ = []
                for i in range(margs.virtual_pipeline_model_parallel_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    # Set pre_process and post_process only after virtual rank is set.
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    this_model = model_provider(
                        pre_process=pre_process,
                        post_process=post_process
                    ).to(dtype)
                    model_.append(this_model)
            else:
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                model_rank = 0
                model_ = [model_provider(pre_process, post_process).to(dtype)]
            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            load_checkpoint_mg(model_, None, None)

            if consumed_train_samples is not None:
                if margs.consumed_train_samples != consumed_train_samples:
                    return None
            else:
                consumed_train_samples = margs.consumed_train_samples
            if consumed_valid_samples is not None:
                if margs.consumed_valid_samples != consumed_valid_samples:
                    return None
            else:
                consumed_valid_samples = margs.consumed_valid_samples
            for vp_rank in range(model_array_len):
                models[vp_rank].append(model_[vp_rank])
        return models

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)

    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        vb_file = open(args.vocab_file)
        vocab = json.load(vb_file)
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print("Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
            queue.put("exit")
        vb_file.close()
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Layernorm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        norm_has_bias = checkpoint_args.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    # metadata
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
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args
    md.embed_layernorm = margs.embed_layernorm

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    all_models = [get_models(tp_size, md.params_dtype)]
    models = all_models[0][0]

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples
    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = {
        "word embeddings": torch.cat(
            [models[tp_rank].language_model.embedding.word_embeddings.weight.data for tp_rank in range(tp_size)],
            dim=0)
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = models[0].language_model.embedding.position_embeddings.weight.data
    if md.embed_layernorm:
        message["word embeddings norm_w"] = models[0].language_model.embedding.word_embeddings.norm.weight.data
        message["word embeddings norm_b"] = models[0].language_model.embedding.word_embeddings.norm.bias.data
    queue_put("embeddings", message)

    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            if pp_rank > 0:
                mpu.set_pipeline_model_parallel_rank(pp_rank)
                if vp_rank == 0:
                    all_models.append(get_models(tp_size, md.params_dtype))
            models = all_models[pp_rank][vp_rank]
            for layer_num, _ in enumerate(models[0].language_model.encoder.layers):
                message = {}

                # Get non-parallel tensors from tp_rank 0
                layer = models[0].language_model.encoder.layers[layer_num]
                message["input norm weight"] = layer.input_norm.weight.data
                if norm_has_bias:
                    message["input norm bias"] = layer.input_norm.bias.data
                message["post norm weight"] = layer.post_attention_norm.weight.data
                if norm_has_bias:
                    message["post norm bias"] = layer.post_attention_norm.bias.data
                if md.linear_bias:
                    message["dense bias"] = layer.self_attention.dense.bias.data
                    message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data
                if args.add_dense_bias:
                    message["dense bias"] = layer.self_attention.dense.bias.data

                # Grab all parallel tensors for this layer
                qkv_weight = []
                qkv_bias = []
                dense_weight = []
                mlp_l0_weight = []
                mlp_l0_bias = []
                mlp_l1_weight = []
                for tp_rank, model in enumerate(models):
                    layer = model.language_model.encoder.layers[layer_num]
                    qkv_weight.append(layer.self_attention.query_key_value.weight.data)
                    dense_weight.append(layer.self_attention.dense.weight.data)
                    mlp_l0_weight.append(layer.mlp.dense_h_to_4h.weight.data)
                    mlp_l1_weight.append(layer.mlp.dense_4h_to_h.weight.data)
                    if md.linear_bias:
                        qkv_bias.append(layer.self_attention.query_key_value.bias.data)
                        mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)
                    if args.add_qkv_bias:
                        qkv_bias.append(layer.self_attention.query_key_value.bias.data)

                # Handle gated linear units
                if md.swiglu:
                    # concat all the first halves ('W's) and all the second halves ('V's)
                    for tp_rank in range(tp_size):
                        mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
                    message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                    message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
                else:
                    message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

                # simple concat of the rest
                message["qkv weight"] = torch.cat(qkv_weight, dim=0)
                message["dense weight"] = torch.cat(dense_weight, dim=1)
                message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
                if md.linear_bias:
                    message["qkv bias"] = torch.cat(qkv_bias, dim=0)
                    if md.swiglu:
                        for tp_rank in range(tp_size):
                            mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                        message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                        message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
                    else:
                        message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
                if args.add_qkv_bias:
                    message["qkv bias"] = torch.cat(qkv_bias, dim=0)

                queue_put(f"transformer layer {total_layer_num}", message)

                total_layer_num = total_layer_num + 1

    # Send final norm from tp_rank 0
    message = {
        "weight": models[0].language_model.encoder.final_norm.weight.data,
    }
    if norm_has_bias:
        message["bias"] = models[0].language_model.encoder.final_norm.bias.data
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": torch.cat(
                [models[tp_rank].language_model.output_layer.weight.data for tp_rank in range(tp_size)],
                dim=0)
        }
        queue_put("output layer", message)

    # Send BERT lm head and binary head if it exists
    if md.model_type == 'BERT':
        message = {
            "weight": models[0].language_model.pooler.dense.weight.data,
            "bias": models[0].language_model.pooler.dense.bias.data,
        }
        queue_put("pooler", message)

        message = {
            "dense weight": models[0].lm_head.dense.weight.data,
            "dense bias": models[0].lm_head.dense.bias.data,
            "norm weight": models[0].lm_head.norm.weight.data,
        }
        if norm_has_bias:
            message["norm bias"] = models[0].lm_head.norm.bias.data
        queue_put("lm head", message)

        if md.bert_binary_head:
            message = {
                "weight": models[0].binary_head.weight.data,
                "bias": models[0].binary_head.bias.data,
            }
            queue_put("binary head", message)
    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
