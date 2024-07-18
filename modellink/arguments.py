# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

import os
from functools import wraps
import argparse
import torch
from megatron.training import print_rank_0
from megatron.training.arguments import load_retro_args, _print_args, _check_arg_is_not_none


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def parse_args_decorator(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'
    parser = _add_network_size_args(parser)
    parser = _add_lora_args(parser)
    parser = _add_data_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_num_layer_allocation(parser)
    parser = _add_profile_args(parser)
    parser = _add_network_args(parser)
    parser = _add_training_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_algorithm_args(parser)
    parser = _add_alibi_args(parser)
    parser = _add_dataset_args(parser)
    parser = _add_high_availability_args(parser)
    return parser


def _add_profile_args(parser):
    group = parser.add_argument_group(title='profiler')

    group.add_argument('--profile-level', type=str, default='level0',
                       choices=['level0', 'level1', 'level2'], help='profiling level0, level1, level2')
    group.add_argument('--profile-with-stack', action='store_true', help='profiling with stack info')
    group.add_argument('--profile-with-memory', action='store_true', help='profiling with memory info')
    group.add_argument('--profile-record-shapes', action='store_true', help='profiling with shape info')
    group.add_argument('--profile-with-cpu', action='store_true', help='profiling with cpu info')
    group.add_argument('--profile-save-path', type=str, default='./profile_dir',
                       help='path to save profiling files')

    return parser


def _add_lora_args(parser):
    group = parser.add_argument_group(title='lora')

    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Lora target modules.')
    group.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    group.add_argument('--lora-r', type=int, default=16,
                       help='Lora r.')
    group.add_argument('--lora-alpha', type=int, default=32,
                       help='Lora alpha.')
    group.add_argument('--lora-modules-to-save', nargs='+', type=str, default=None,
                       help='Lora modules to save.')
    group.add_argument('--lora-register-forward-hook', nargs='+', type=str,
                       default=['word_embeddings', 'input_layernorm'],
                       help='Lora register forward hook.')

    return parser


def _add_moe_args(parser):
    group = parser.add_argument_group(title='moe')
    group.add_argument('--moe-router-topk', type=int, default=2,
                       help='Number of experts to route to for each token. The default is 2.')
    group.add_argument('--moe-router-load-balancing-type', type=str,
                       choices=['aux_loss'],
                       default='aux_loss',
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds '
                            'to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds '
                            'to the balancing algorithm used in S-BASE, "softmax_topk" implies no load balancing and '
                            'softmax before topk ,and "None" implies no load balancing. '
                            'The default is "aux_loss".')
    group.add_argument('--expert-interval', type=int, default=1,
                       help='Use experts in every "expert-interval" layers')
    group.add_argument('--moe-aux-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.')
    group.add_argument('--moe-z-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.')
    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time')
    group.add_argument('--noisy-gate-policy', type=str, default=None,
                       help="noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.")
    group.add_argument('--enable-token-rearrange-opt', action='store_true',
                       help="Use this flag to enable token rearrange optimize")
    group.add_argument('--embedding-multiplier-scale', type=float, default=1.0,
                       help='add scale for embedding.')
    group.add_argument('--input-jitter', action='store_false', help='Add noise to the input tensor.')
    group.add_argument('--post-norm', action='store_true', help='post norm after attention or mlp.')
    group.add_argument('--output-multiplier-scale', type=float, default=None, help='Add scale for logits output.')
    group.add_argument("--moe-permutation-async-comm", action='store_true',
                       help="overlap moe permutation 3 all gather communications")
                       
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data')
    group.add_argument('--is-instruction-dataset', action='store_true', help='use instruction dataset or not')
    group.add_argument('--variable-seq-lengths', action='store_true', help='Use variable seq lengths or not.')
    group.add_argument("--tokenizer-kwargs", type=str, nargs='+', default=None,
                       help="Kwargs of the huggingface tokenizer.")
    group.add_argument('--tokenizer-padding-side', type=str, default='right',
            help="tokenizer padding side")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'Llama2Tokenizer',
                                'PretrainedFromHF',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")

    return parser


def _add_num_layer_allocation(parser):
    group = parser.add_argument_group(title='num_layer_allocation')
    group.add_argument('--num-layer-list',
                       type=str, help='a list of number of layers, '
                                'seperated by comma; e.g., 4,4,4,4')
    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network_size_args')
    group.add_argument('--padded-vocab-size',
                       type=int,
                       default=None,
                       help='set padded vocab size')
    group.add_argument('--embed-layernorm',
                       action='store_true',
                       default=False,
                       help='set padded vocab size'
                       )
    group.add_argument('--use-partial-rope',
                       action='store_true',
                       help='use partial rope in ChatGLM3.'
                       )
    
    group.add_argument("--use-fused-rmsnorm", action='store_true',
                       help="Use fused rmsnorm.")
    group.add_argument("--use-fused-swiglu", action='store_true',
                       help="Use fused swiglu.")
    group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                       help="Use fused rotary-pos-emb.")
    group.add_argument("--use-mc2", action='store_true',
                       help="Use mc2 for compute-comm overlap in tp.")
    group.add_argument('--sliding-window', type=int, default=None,
                       help='Window size when use sliding window attention.')
    return parser


def _add_algorithm_args(parser):
    group = parser.add_argument_group(title='algorithm')
    group.add_argument('--rotary-base', type=float, help='rotary-base.')

    return parser


def _add_network_args(parser):
    group = parser.add_argument_group(title='network')

    group.add_argument("--add-qkv-bias", action="store_true", default=False,
                       help='Configuration for the qkv bias.')
    group.add_argument("--add-dense-bias", action="store_true", default=False,
                       help='Configuration for the dense bias.')
    group.add_argument("--skip-bias-add", action="store_false", default=True,
                       help='Configuration for the skip bias.')
    group.add_argument('--add-rmsnorm-offset', action='store_true', default=False,
                       help='RMSNorm unit offset.')
    group.add_argument('--geglu', action='store_true', default=False,
                       help='Geglu activate function.')
    group.add_argument('--input-embeds-norm', action='store_true', default=False,
                       help='input normalization.')
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    # gradient_accumulation_fusion保持常闭
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false', default=False,
                       help='Disable fusing gradient accumulation to weight '
                            'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    # transformer-impl保持local
    group.add_argument('--transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--pre-tockens', type=int, default=65536,
                       help='pre-tockens is used by Flash attention')
    group.add_argument('--next-tockens', type=int, default=0,
                       help='next-tockens is used by Flash attention')
    group.add_argument('--shape-order', type=str, default='SBH',
                       choices=['SBH', 'BSH', 'BSND'],
                       help='input shape order used by Flash attention')
    group.add_argument('--use-deter-comp',
                       action='store_true',
                       default=False,
                       help='enable deterministic computing for npu')
    group.add_argument('--jit-compile', action='store_true', default=False,
                       help='Setting jit compile mode to True')
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--local-rank', type=int, default=None,
                       help='Local rank passed from distributed launcher for torch2.x.')
    return parser


def add_parser_argument_choices_value(parser, argument_name, value):
    if parser._actions:
        for action in parser._actions:
            if isinstance(action, argparse._ArgumentGroup):
                add_parser_argument_choices_value(action, argument_name)
            elif isinstance(action, argparse.Action) and argument_name in action.option_strings:
                action.choices.append(value)


def _add_alibi_args(parser):
    add_parser_argument_choices_value(parser, "--position-embedding-type", 'alibi')

    group = parser.add_argument_group(title='alibi')
    group.add_argument('--square-alibi-mask',
                       action='store_true',
                       default=False,
                       help='attention mask of alibi is squared')
    group.add_argument('--fill-neg-inf',
                       action='store_true',
                       default=False,
                       help='fill alibi with negative inf')

    return parser


def _add_high_availability_args(parser):
    group = parser.add_argument_group(title='high_availability')

    group.add_argument('--enable-high-availability',
                       action='store_true',
                       help='switch of the high availability feature')

    group.add_argument('--enable-optimizer-state-local-copy',
                       action='store_true',
                       help='high availability feature, enable parameter state local copy of distributed optimizer')

    return parser


def _add_dataset_args(parser):
    group = parser.add_argument_group(title='dataset_args')
    group.add_argument('--no-shared-storage',
                       action='store_true',
                       default=False,
                       help='if no shared storage, set it'
                       )
    return parser


def _validate_create_attention_mask_in_dataloader(args):
    args.create_attention_mask_in_dataloader = False
    reset_data = args.reset_attention_mask or args.reset_position_ids
    alibi_without_flash_attn = args.position_embedding_type == 'alibi' and not args.use_flash_attn
    if (reset_data or alibi_without_flash_attn or args.tokenizer_padding_side == "left"):
        args.create_attention_mask_in_dataloader = True
    print_rank_0("create-attention-mask-in-dataloader is {}".format(args.create_attention_mask_in_dataloader))


def _validate_position_embedding(args):
    """
    validate position embedding arguments.
    """
    if args.use_partial_rope and args.use_fused_rotary_pos_emb:
        raise AssertionError('Fused rotary embedding is not supported in partial rope.')
    if args.position_embedding_type == 'alibi' and args.sliding_window is not None:
        raise AssertionError('Sliding Window Attention is forbidden when use alibi.')
    if args.tokenizer_padding_side == 'left' and args.position_embedding_type == 'alibi':
        raise AssertionError('Alibi is not support tokenizer-padding-side left now.')


def _validate_high_availability(args):
    if args.enable_optimizer_state_local_copy and not args.enable_high_availability:
        raise AssertionError('switch of the high availability feature is unsupported')


def _validate_instruction_finetune(args):
    if args.variable_seq_lengths:
        if args.context_parallel_size > 1:
            raise AssertionError('Context parallelism is forbidden when use variable seq lengths.')


def validate_args(args, defaults={}):
    # Load saved args from Retro (if applicable).
    load_retro_args(args)

    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)

    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )

    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    assert args.world_size % (model_parallel_size * args.context_parallel_size) == 0, \
        'world size ({}) is not divisible by tensor parallel size ({}) times ' \
        'pipeline parallel size ({}) times context parallel size ({})'.format(
        args.world_size, args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size, args.context_parallel_size)
    args.data_parallel_size = args.world_size // (model_parallel_size * args.context_parallel_size)
    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {} '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)
    if args.pipeline_model_parallel_size > 1:
        if args.pipeline_model_parallel_split_rank is not None:
            assert args.pipeline_model_parallel_split_rank < \
                    args.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.pipeline_model_parallel_size)

    if args.tp_comm_overlap:
        assert args.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # Deprecated arguments
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    if args.checkpoint_activations:
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
            'number of layers should be divisible by the pipeline parallel size'
        num_layers_per_pipeline_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
        assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
        args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        if args.rank == 0:
            print('WARNING: Setting args.overlap_p2p_comm to False since non-interleaved '
                  'schedule does not support overlapping p2p communication')

    if args.overlap_param_gather:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            '--overlap-grad-reduce should be turned on when using --overlap-param-gather'
        assert args.use_mcore_models, \
            '--overlap-param-gather only supported with MCore models'

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.num_layers
    else:
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length
    if args.decoder_seq_length is not None:
        assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.moe_grouped_gemm:
        assert args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
        dc = torch.cuda.get_device_capability()
        assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert (TORCH_MAJOR, TORCH_MINOR) >= (1, 10), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
        if args.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Retro checks.
    if args.retro_add_retriever:

        # Train samples should be auto-loaded.
        assert args.train_samples is not None, \
            "args.train_samples should be auto-loaded from the retro config."

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        assert args.use_mcore_models, \
            '--decoupled-lr and --decoupled-min-lr only supported by Megatron Core, please add --use-mcore-models.'

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and not args.use_mcore_models:
        raise RuntimeError('--rotary-interleaved only support Megatron Core, please add --use-mcore-models.')

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # MoE Spec check
    if args.num_experts is not None:
        assert args.spec is None, "Model Spec must be None when using MoEs"
        if args.tensor_model_parallel_size > 1:
            assert args.sequence_parallel, \
                "When using MoE and tensor parallelism, sequence parallelism must be used."

    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.fp16, \
            "Expert parallelism is not supported with fp16 training."

    # Distributed checkpointing checks
    if args.use_dist_ckpt and not args.use_mcore_models:
        raise RuntimeError('--use-dist-ckpt only support Megatron Core, please add --use-mcore-models.')

    _validate_create_attention_mask_in_dataloader(args)
    _validate_instruction_finetune(args)
    _validate_position_embedding(args)
    _validate_high_availability(args)

    # Print arguments.
    _print_args("arguments", args)

    return args