# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import wraps
import argparse


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def parse_args_wrapper(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'
    parser = _add_network_size_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_training_args(parser)
    parser = _add_data_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_cp_args(parser)
    parser = _add_network_args(parser)
    parser = _add_algorithm_args(parser)
    parser = _add_automated_pipeline_args(parser)
    parser = _add_alibi_args(parser)

    return parser


def _add_moe_args(parser):
    group = parser.add_argument_group(title='moe')
    # deepspeed moe arguments
    group.add_argument('--moe-model-type', type=str, default='megatron_moe',
                       choices=['deepspeed_moe', 'megatron_moe'], help='moe model type default megatron moe')
    group.add_argument('--expert-interval', type=int, default=1,
                       help='Use experts in every "expert-interval" layers')
    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time')
    group.add_argument('--noisy-gate-policy', type=str, default=None,
                       help="noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.")
    group.add_argument('--enable-token-rearrange-opt', action='store_true',
                       help="Use this flag to enable token rearrange optimize")
    group.add_argument('--no-use-rts',
                       action='store_false', default=False,
                       help='whether to use Random Token Selection.',
                       dest='use_rts')
    group.add_argument("--moe-no-drop", action='store_true',
                       help="Use no drop policy in moe layer, no tokens will be discarded.")
    group.add_argument("--moe-dynamic-padding", action='store_true',
                       help="Reducing AllReduce communication under the no drop policy through the sliding window mechanism.")
    group.add_argument("--moe-use-sinkhorn", action='store_true',
                       help="Use sinkhorn load balancing in the gate.")

    # megatron mcore moe arguments
    group.add_argument("--moe-permutation-async-comm", action='store_true',
                       help="overlap moe permutation 3 all gather communications")
    return parser


def _add_cp_args(parser):
    group = parser.add_argument_group(title='cp parallel')
    group.add_argument('--context-parallel-algo', type=str, default='ulysses_cp_algo',
                       choices=['ulysses_cp_algo', 'megatron_cp_algo', 'hybrid_cp_algo'], help='context parallel algorithm')
    group.add_argument('--ulysses-degree-in-cp', type=int, default=None)
    group.add_argument('--cp-attention-mask-type', type=str, default='causal',
                       choices=['causal', 'full'], help='context parallel attention mask type')
    group.add_argument('--use-cp-send-recv-overlap', action='store_true',
                       help='use this flag to enable cp send-recv-overlap.')
    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')
    group.add_argument("--use-fused-rmsnorm", action='store_true',
                       help="Use fused rmsnorm.")
    group.add_argument("--use-fused-swiglu", action='store_true',
                       help="Use fused swiglu.")
    group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                       help="Use fused rotary-pos-emb.")
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')
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


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--local-rank', type=int, default=None,
                       help='Local rank passed from distributed launcher for torch2.x.')
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    # gradient_accumulation_fusion保持常闭
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false', default=False,
                       help='Disable fusing gradient accumulation to weight '
                            'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--pre-tockens', type=int, default=65536,
                       help='pre-tockens is used by Flash attention')
    group.add_argument('--next-tockens', type=int, default=0,
                       help='next-tockens is used by Flash attention')
    group.add_argument('--shape-order', type=str, default='SBH',
                       choices=['SBH', 'BSH', 'BSND'],
                       help='input shape order used by Flash attention')
    group.add_argument('--sparse-mode', type=int, default=0,
                       help='To improve performance in different modes of attention mask')
    group.add_argument('--adaptive-recompute-device-size',
                       type=int, default=-1,
                       help='The memory size for adaptive selective recompute strategy. '
                            'The default is -1. If this parameter > 0, '
                            'will activate adaptive selective recompute. ')
    group.add_argument('--adaptive-recompute-profiling-step',
                       type=int, default=10,
                       help='The profiling step for adaptive selective recompute strategy. '
                            'The default is 10. If activate adaptive selective recompute, '
                            'will solve graph after step 10. ')
    group.add_argument('--adaptive-recompute-device-swap',
                       action='store_true', default=False,
                       help='switch to open adaptive recompute feature. '
                            'The default is False.')
    group.add_argument('--recompute-activation-function', action='store_true',
                       help='Recompute the activation function in MLP layers.')
    group.add_argument('--recompute-activation-function-num-layers', type=int, default=None,
                       help='Can be used together with "--recompute-method block." '
                       'and "--recompute-num-layers". ')
    group.add_argument('--jit-compile', action='store_true', default=False,
                       help='Setting jit compile mode to True')
    group.add_argument('--use-fusion-attn-v2', action='store_true', default=False,
                       help='use fusion_attention ops version 2')
    return parser


def _add_network_args(parser):
    group = parser.add_argument_group(title='network')

    group.add_argument("--add-qkv-bias", action="store_true", default=False,
                       help='Configuration for the qkv bias.')
    group.add_argument("--add-dense-bias", action="store_true", default=False,
                       help='Configuration for the dense bias.')
    group.add_argument("--skip-bias-add", action="store_false", default=True,
                       help='Configuration for the skip bias.')
    return parser


def _add_automated_pipeline_args(parser):
    group = parser.add_argument_group(title='automated_pipeline_allocation')
    group.add_argument('--automated-pipeline',
                       action='store_true',
                       help='To enable automated pipeline process'
                      )
    group.add_argument('--save-memory-ratio',
                       type=float, default=0.20,
                       help='To set memory saving rate in automated pipeline'
                       )
    group.add_argument('--num-layer-list',
                       type=str, help='To store the layer policy of automated pipeline'
                       )
    group.add_argument('--recompute-module-list',
                       type=str, help='To store the recompute policy of automated pipeline'
                       )
    group.add_argument('--recompute-type',
                       type=int, default=2,
                       help='To store the recompute type of automated pipeline, 0 for mlp block '
                       '1 for attention block and 2 for transformer layer'
                       )
    return parser


def _add_algorithm_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--reuse-fp32-param', action='store_true',
                       help='The distributed training optimizer frees up '
                            'param copies of FP32 to save memory.')
    group.add_argument('--rotary-base', type=float, help='rotary-base.')

    group.add_argument('--optimize-recomp-communication-level', type=int, default=0,
                       help='The algorithm optimize the level of tp communication in the recompute stage.')
    group.add_argument('--optimize-send-recv-comm', action='store_true', 
                       help='optimize send_recv communication in pipeline without interleaving.')
    return parser


def core_transformer_config_from_args_wrapper(fn):
    @wraps(fn)
    def wrapper(args):
        config = fn(args)
        config.context_parallel_algo = args.context_parallel_algo
        config.batch_p2p_comm = False
        return config

    return wrapper


def validate_args_wrapper(validate_args):
    @wraps(validate_args)
    def wrapper(args, defaults):
        overlap_param_gather_without_mcore_models = False
        if args.overlap_param_gather and not args.use_mcore_models:
            args.use_mcore_models = True
            overlap_param_gather_without_mcore_models = True

        args = validate_args(args, defaults)
        if args.use_fused_rmsnorm:
            if args.normalization != "RMSNorm":
                raise AssertionError(
                    '--use-fused-rmsnorm must enable with '
                    '--normalization=RMSNorm, but got normalization'
                    '={}.'.format(args.normalization))
        if args.use_fused_swiglu:
            if not args.swiglu:
                raise AssertionError(
                    '--use-fused-swiglu must enable with --swiglu, '
                    'but --swiglu={}.'.format(args.swiglu))
        if args.use_fused_rotary_pos_emb:
            if args.position_embedding_type != 'rope':
                raise AssertionError(
                    '--use-fused-rotary-pos-emb must enable with'
                    '--position-embedding-type=rope')
        if args.alibi_fusion_attn_type is not None and args.alibi_fusion_attn_type not in [0, 2, 3]:
            raise AssertionError('--alibi-fusion-attn-type only support for `0, 2, 3`')
        if args.reuse_fp32_param and not args.bf16:
            raise AssertionError('--reuse-fp32-param only support for `bf16`')
        if args.optimize_recomp_communication_level > 0:
            if not hasattr(args, "optimize_recomp_communication_status"):
                args.optimize_recomp_communication_status = 0
        if args.moe_dynamic_padding and not args.moe_no_drop:
            raise AssertionError('`--moe-dynamic-padding` only support for `--moe-no-drop`.')
        if args.moe_permutation_async_comm and args.moe_model_type != 'megatron_moe':
            raise AssertionError('`--moe-permutation-async-comm` only support for megatron core moe.')

        if args.context_parallel_size > 1 and args.context_parallel_algo == 'ulysses_cp_algo':
            head, remainder = divmod(args.num_attention_heads, args.context_parallel_size * args.tensor_model_parallel_size)
            assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by context_parallel_size * tensor_model_parallel_size"
            args.use_flash_attn = True
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'megatron_cp_algo':
            assert args.seq_length % (2 * args.context_parallel_size) == 0, f"sequence length must be divisible by 2 * context_parallel_size"
            args.use_flash_attn = True
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'hybrid_cp_algo':
            assert args.ulysses_degree_in_cp is not None, "--ulysses-degree-in-cp must be specified in hybrid_cp_algo"
            ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
            assert ring_degree > 1 and remainder == 0, "--ulysses-degree-in-cp must be devisible by --context-parallel-size"

            head, remainder = divmod(args.num_attention_heads, args.ulysses_degree_in_cp * args.tensor_model_parallel_size)
            assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp"

            assert args.seq_length % (2 * args.context_parallel_size) == 0, f"sequence length must be divisible by 2 * context_parallel_size in hybrid cp"
            args.use_flash_attn = True
            
            
        # Mandatory modification to SBH, subsequent abandonment of other formats such as BSH,BSND
        if args.shape_order != 'SBH':
            args.shape_order = 'SBH'
        if overlap_param_gather_without_mcore_models:
            args.use_mcore_models = False
        if args.transformer_impl == 'transformer_engine':
            args.transformer_impl = 'local'
        if args.fp8:
            raise AssertionError('NPU not supported FP8.')
        if args.tp_comm_overlap:
            args.tp_comm_overlap = False
        if args.recompute_method == "uniform":
            assert not args.recompute_activation_function, \
                'uniform recomputation is not compatible ' \
                'with activation function recomputation '
        if args.recompute_activation_function and args.recompute_granularity == "selective":
            raise AssertionError('--recompute-activation-function is not compatible with selective recomputation')
        adaptive_recompute_enable = args.adaptive_recompute_device_size > 0 or args.adaptive_recompute_device_swap
        if adaptive_recompute_enable and (args.recompute_granularity is not None or args.recompute_method is not None):
            print("[WARNING] if activate adaptive selective recompute, recomputing args switch "
                        "\"recompute_granularity\" and \"recompute_method\" will be turned off.")
            args.recompute_granularity = None
            args.recompute_method = None
        if args.use_flash_attn:
            assert args.sparse_mode == 0 or args.sparse_mode == 2, f"Only supports sparse modes 0 and 2"
        args.create_attention_mask_in_dataloader = False

    return wrapper


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

    group.add_argument('--alibi-fusion-attn-type',
                    type=int,
                    help='alibi pse type, support for 0,2,3')

    group.add_argument('--alibi-diagonal-opposite',
                       action='store_true',
                       default=False,
                       help='make alibi diagonal opposite')

    return parser
