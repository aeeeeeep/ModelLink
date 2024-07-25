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
from modellink.utils import print_rank0_by_args


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
    parser = _add_cp_args(parser)

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


def _add_cp_args(parser):
    group = parser.add_argument_group(title='cp parallel')
    group.add_argument('--context-parallel-algo', type=str, default='ulysses_cp_algo',
                       choices=['ulysses_cp_algo', 'megatron_cp_algo', 'hybrid_cp_algo'], help='context parallel algorithm')
    group.add_argument('--ulysses-degree-in-cp', type=int, default=None)
    group.add_argument('--cp-attention-mask-type', type=str, default='causal',
                       choices=['causal', 'full'], help='context parallel attention mask type')
    group.add_argument('--use-cp-send-recv-overlap', action='store_true',
                       help='use it to enable cp send-recv-overlap.')
    group.add_argument('--kv-head-repeat-before-uly-alltoall', action='store_true', default=True,
                       help='use it to expand key and value for ulysses when GQA/MQA is used.')
    return parser


def _validate_cp_args(args):
    def _check_attention_head(args, uly_size):
        """
        check GQA & ulysses
        """
        head, remainder = divmod(args.num_attention_heads, uly_size * args.tensor_model_parallel_size)
        assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by ulysses_size * tensor_model_parallel_size"
        if args.group_query_attention and args.num_query_groups >= 1:
            head_split_by_tp, remainder = divmod(args.num_query_groups, args.tensor_model_parallel_size)
            assert head_split_by_tp >= 1 and remainder == 0, f"num_query_groups must be divisible by tensor_model_parallel_size"

            if not args.kv_head_repeat_before_uly_alltoall:
                head_split_by_tp_cp, remainder = divmod(head_split_by_tp, uly_size)
                if not (head_split_by_tp_cp >= 1 and remainder == 0):
                    raise AssertionError(
                        'num_query_groups must be divisible by ulysses_size * tensor_model_parallel_size.\n'
                        'Solution 1. adjust the ulysses_size\n'
                        'Solution 2. You can enable --kv-head-repeat-before-uly-alltoall to roll on.\n'
                        'However, performance would be affected since it would increase communication volume \n'
                        'for ulysses alltoall as well as memory usage.')

    if args.context_parallel_size <= 1:
        return

    # In context parallel we use FA
    args.use_flash_attn = True
    print_rank0_by_args(args, f"[INFO] Setting args.use_flash_attn={args.use_flash_attn} since context parallel is enabled.")
    if not args.use_mcore_models:
        raise AssertionError(f"Context parallel is only supported in Mcore.")

    if args.context_parallel_algo == 'ulysses_cp_algo':
        assert args.seq_length % args.context_parallel_size == 0, f"sequence length must be divisible by context_parallel_size"
        head, remainder = divmod(args.num_attention_heads,
                                 args.context_parallel_size * args.tensor_model_parallel_size)
        assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by context_parallel_size * tensor_model_parallel_size"
    if args.context_parallel_algo == 'megatron_cp_algo':
        assert args.seq_length % (
                    2 * args.context_parallel_size) == 0, f"sequence length must be divisible by 2 * context_parallel_size"
        _check_attention_head(args, args.context_parallel_size)

    if args.context_parallel_algo == 'hybrid_cp_algo':
        assert args.ulysses_degree_in_cp is not None, "--ulysses-degree-in-cp must be specified in hybrid_cp_algo"
        ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
        assert ring_degree > 1 and remainder == 0, "--ulysses-degree-in-cp must be devisible by --context-parallel-size"

        head, remainder = divmod(args.num_attention_heads,
                                 args.ulysses_degree_in_cp * args.tensor_model_parallel_size)
        assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp"
        assert args.seq_length % (
                    2 * args.context_parallel_size) == 0, f"sequence length must be divisible by 2 * context_parallel_size in hybrid cp"
        _check_attention_head(args, args.ulysses_degree_in_cp)

    if args.sliding_window:
        raise AssertionError("sliding window is not supported in context parallel.")


def _validate_tocken(args):
    """To avoid invalid tocken configration."""
    if args.pre_tocken > args.seq_length:
        print_rank0_by_args(args, f"[INFO] pre_tocken={args.pre_tocken} would be adjusted to {args.seq_length} for better performance.")
    if args.next_tocken > args.seq_length:
        print_rank0_by_args(args, f"[INFO] next_tocken={args.next_tocken} would be adjusted to {args.seq_length} for better performance.")


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
    group.add_argument('--sparse-mode', type=int, default=0,
                       help='different modes of flash attention mask')
    group.add_argument('--shape-order', type=str, default='SBH',
                       choices=['SBH', 'BSH', 'BSND'],
                       help='input shape order used by Flash attention')
    group.add_argument('--use-deter-comp',
                       action='store_true',
                       default=False,
                       help='enable deterministic computing for npu')
    group.add_argument('--jit-compile', action='store_true', default=False,
                       help='Setting jit compile mode to True')
    group.add_argument('--prompt-type', type=str, default=None,
                       choices=['default', 'empty', 'chatglm2', 'chatglm3', 'chatglm3_system', 'chatml', 'chatml_de', 'qwen', 'llama3', 'mistral', 'mixtral'],
                       help='Which template to use for constructing prompts in training/inference/evaluation.'
                            'e.g., "qwen"')

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
    if reset_data or alibi_without_flash_attn or args.tokenizer_padding_side == "left":
        args.create_attention_mask_in_dataloader = True
    print_rank0_by_args(args, f"[INFO] Setting args.create_attention_mask_in_dataloader to {args.create_attention_mask_in_dataloader} "
                 f"since reset_data={reset_data} or alibi_without_flash_attn={alibi_without_flash_attn} or "
                 f"args.tokenizer_padding_side={args.tokenizer_padding_side}")


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


def validate_args_decorator(megatron_validate_args):
    @wraps(megatron_validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}
        variable_seq_lengths = args.variable_seq_lengths
        megatron_validate_args(args, defaults)
        args.variable_seq_lengths = variable_seq_lengths

        args.use_mc2 = False

        _validate_cp_args(args)
        _validate_create_attention_mask_in_dataloader(args)
        _validate_instruction_finetune(args)
        _validate_position_embedding(args)
        _validate_high_availability(args)

        from modellink.utils import print_args
        print_args('ModelLink Arguments', args)
        return args

    return wrapper

