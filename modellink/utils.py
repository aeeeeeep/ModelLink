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

"""General utilities."""
import os
import stat
import random

from functools import wraps

import torch
import torch_npu
import numpy as np

import megatron
from megatron.training import get_args
from megatron.core import mpu


WRITE_FILE_DEFAULT_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_FILE_DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR


def parse_args():
    return megatron.training.arguments.parse_args()


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
                torch.distributed.get_rank() % torch.cuda.device_count() == 0
        ):
            return True
        else:
            return False
    else:
        return True


def get_tune_attention_mask(attention_mask_1d):
    args = get_args()
    micro_batch_size, seq_length = attention_mask_1d.size()
    if args.reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = None

    if args.tokenizer_padding_side == "left":
        attention_mask = torch.tril(torch.ones(seq_length, seq_length, device=attention_mask_1d.device, dtype=torch.bool)).view(1, 1, seq_length, seq_length)
        attention_mask_tran = attention_mask_1d.view(seq_length, 1, -1)
        attention_mask = attention_mask.masked_fill((attention_mask_tran < 0.5).view(-1, 1, 1, seq_length), value=0)
    else:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=attention_mask_1d.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    attention_mask = attention_mask.masked_fill((attention_mask_1d < 0.5).view(-1, 1, 1, seq_length), value=0)
    attention_mask = (attention_mask < 0.5)
    return attention_mask


def print_args_wrapper(fn):
    """
    Add switch for controlling when to print arguments.
    """
    @wraps(fn)
    def wrapper(title, args, after_validate=False):
        if after_validate:
            fn(title, args)

    return wrapper


def print_args(title, args):
    """
    Provide a public func for printing arguments.
    """
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------',
              flush=True)


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def unwrap_model_wrapper(fn):
    @wraps(fn)
    def wrapper(model, module_instances=None):
        if not module_instances:
            module_instances = megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES
        return fn(model, module_instances)
    return wrapper


def get_finetune_data_on_this_tp_rank(data_iterator):
    args = get_args()
    ds = next(data_iterator)
    tokens = ds.get('input_ids').long().cuda(non_blocking=True)
    attention_mask = None

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        via_length = torch.LongTensor([tokens.shape[1]]).cuda(non_blocking=True)
        _broadcast(via_length)
        _broadcast(tokens)
        if args.position_embedding_type == "alibi":
            attention_mask_1d = ds.get('attention_mask').long().cuda(non_blocking=True)
            _broadcast(attention_mask_1d)
            attention_mask = get_tune_attention_mask(attention_mask_1d)
    else:
        via_length = torch.empty((1), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(via_length)
        tokens = torch.empty((args.micro_batch_size, via_length), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(tokens)
        if args.position_embedding_type == "alibi":
            attention_mask_1d = torch.empty((args.micro_batch_size, via_length), dtype=torch.int64, device=torch.cuda.current_device())
            _broadcast(attention_mask_1d)
            attention_mask = get_tune_attention_mask(attention_mask_1d)

    return tokens, attention_mask