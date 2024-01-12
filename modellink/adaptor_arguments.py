from functools import wraps
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
        args =  parse_args(decorated_provider, ignore_unknown_args)
        if args.use_alibi_position_embedding:
             args.position_embedding_type = "alibi"
        return args

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'
    parser = _add_lora_args(parser)
    parser = _add_alibi_args(parser)
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


def _add_alibi_args(parser):
    group = parser.add_argument_group(title='alibi')
    group.add_argument('--use-alibi-position-embedding', action='store_true',
                       help='use alibi position embedding ')
    group.add_argument('--square-alibi-mask',
                       action='store_true',
                       default=False,
                       help='attention mask of alibi is squared')
    group.add_argument('--fill-neg-inf',
                       action='store_true',
                       default=False,
                       help='fill alibi with negative inf')

    return parser