# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import sys
import logging
from functools import wraps
import ascendspeed
import megatron


def seq_length_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from megatron import get_args
        self.seq_length = get_args().seq_length
        return fn(*args, **kwargs)

    return wrapper


megatron.model.languange_model.TransformerLanguageModel.forward = (
    seq_length_wrapper(megatron.model.languange_model.TransformerLanguageModel.forward))

from .global_vars import get_args
from .global_vars import get_current_global_batch_size
from .global_vars import get_num_microbatches
from .global_vars import update_num_microbatches
from .global_vars import get_tokenizer
from .global_vars import get_tensorboard_writer
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers
from .global_vars import get_retro_args

from .utils import print_rank_0
from .utils import print_rank_last
from .utils import is_last_rank
from .utils import is_rank_0
