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

import torch

from peft import PeftModel, LoraModel
import megatron
from megatron.core import DistributedDataParallel as DDP
from megatron.model import Float16Module


WRITE_FILE_DEFAULT_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_FILE_DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR

ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module, PeftModel, LoraModel)


def parse_args():
    return megatron.arguments.parse_args()


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


def get_tune_attention_mask(attention_mask_1d, reset_attention_mask=True):
    micro_batch_size, seq_length = attention_mask_1d.size()
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=attention_mask_1d.device)).view(
        att_mask_batch, 1, seq_length, seq_length)
    attention_mask = attention_mask.masked_fill((attention_mask_1d < 0.5).view(-1, 1, 1, seq_length), value=0)
    attention_mask = (attention_mask < 0.5)
    return attention_mask
