from functools import wraps
import argparse

import megatron


def apply_arguments_patch():
    megatron.initialize.parse_args = parse_args_decorator(megatron.initialize.parse_args)
    megatron.arguments.parse_args = parse_args_decorator(megatron.arguments.parse_args)


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
    parser = _add_lora_args(parser)
    parser = _add_data_args(parser)
    parser = _add_alibi_args(parser)
    parser = _add_network_size_args(parser)
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
    group.add_argument('--lora-adapter-name', type=str, default='default',
                       help='Lora adapter name.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data')
    group.add_argument('--is-instruction-dataset', action='store_true', help='use instruction dataset or not')
    group.add_argument('--variable-seq-lengths', action='store_true', help='Use variable seq lengths or not.')

def add_parser_argument_choices_value(parser, argument_name, value):
    for i, action in enumerate(parser._actions):
        if isinstance(action, argparse._ArgumentGroup):
            add_parser_argument_choices_value(action, argument_name)
        elif isinstance(action, argparse.Action) and argument_name in action.option_strings:
            action.choices.append(value)


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network_size_args')
    group.add_argument('padded_vocab_size',
                       action='store_true',
                       default=None,
                       help='set padded vocab size')
    group.add_argument('embed_layernorm',
                       action='store_true',
                       default=True,
                       help='set padded vocab size'
                       )
    return parser


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
