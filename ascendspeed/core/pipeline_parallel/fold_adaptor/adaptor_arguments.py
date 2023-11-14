# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import argparse
import ascendspeed


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--foldx-mode', default=None,
                       choices=['aiao', 'fifo'],
                       help='Choose fold-x pipeline parallelism.')
    group.add_argument('--foldx-dp', action='store_true',
                       help='Gradient sync in foldx process.')    
    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--enable-expert-tensor-parallelism', action='store_true',
                        default=False,
                        help="use tensor parallelism for expert layers in MoE")
    group.add_argument('--sequence-parallel', action='store_true',
                       default=False,
                       help="use sequence parallelism")
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')
    group.add_argument('--moe-expert-parallel-size', type=int, default=1,
                       help='Degree of the MoE expert parallelism.')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'ccl'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='Use scatter/gather to optimize communication of tensors in pipeline',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank passed from distributed launcher.')
    group.add_argument('--local-rank', type=int, default=None,
                       help='Local rank passed from distributed launcher for torch2.x.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.')
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU')
    group.add_argument('--triangle-attn', action='store_true',
                       help="use triangle attention instead self attention")
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')
    return parser

ascendspeed.arguments._add_distributed_args = _add_distributed_args

