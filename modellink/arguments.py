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

from functools import wraps
import argparse
from megatron.training import print_rank_0


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
                       choices=['aux_loss', ],
                       default='aux_loss',
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds '
                            'to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds '
                            'to the balancing algorithm used in S-BASE, and "None" implies no load balancing. '
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


def validate_args_decorator(validate_args):
    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}

        args.create_attention_mask_in_dataloader = False
        reset_data = args.reset_attention_mask == True or args.reset_position_ids == True
        alibi_without_FA = args.position_embedding_type == 'alibi' and not args.use_flash_attn
        if (reset_data or alibi_without_FA or args.tokenizer_padding_side == "left"):
            args.create_attention_mask_in_dataloader = True
        print_rank_0("create-attention-mask-in-dataloader is {}".format(args.create_attention_mask_in_dataloader))

        validate_args(args, defaults)
        if args.position_embedding_type == 'alibi' and args.sliding_window is not None:
            raise AssertionError('Sliding Window Attention is forbidden when use alibi.')
        if args.tokenizer_padding_side == 'left' and args.position_embedding_type == 'alibi':
            raise AssertionError('Alibi is not support tokenizer-padding-side left now.')
        return args

    return wrapper
def _add_dataset_args(parser):
    group = parser.add_argument_group(title='dataset_args')
    group.add_argument('--no-shared-storage',
                       action='store_true',
                       default=False,
                       help='if no shared storage, set it'
                       )
    return parser
