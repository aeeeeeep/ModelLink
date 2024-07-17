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

from megatron.training import get_args
from .ascend_turbo.mc2_linears_seq_parallel import RowSeqParallelLinear


def vocab_embedding_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        output = fn(self, *args, **kwargs)
        args_ = get_args()
        if hasattr(self, 'norm'):
            output = self.norm(output)
        return output * args_.embedding_multiplier_scale if args_.embedding_multiplier_scale else output
    return wrapper


class RowSeqParallelLinearNoComm(RowSeqParallelLinear):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        global_args = get_args()
        world_size = get_tensor_model_parallel_world_size()
        rank = torch.distributed.get_rank(group)
        if global_args.optimize_recomp_communication_status < 2:
            global_args.optimize_recomp_communication_status = global_args.optimize_recomp_communication_status + 1 \
                if global_args.optimize_recomp_communication_status > 0 \
                else global_args.optimize_recomp_communication_status
            return RowSeqParallelLinear.forward(ctx, input_, weight, bias, group)
        else:
            if torch.__version__ > "2.0":
                global_rank = torch.distributed.get_global_rank(group, rank)
                hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                    global_rank
                )
            else:
                hcomm_info = group.get_hccl_comm_name(rank)
            ctx.save_for_backward(input_, weight)
            ctx.hcomm_info = hcomm_info
            ctx.world_size = world_size
            ctx.use_bias = bias is not None
            output_ = torch.matmul(input_, weight.t())
            global_args.optimize_recomp_communication_status = global_args.optimize_recomp_communication_status + 1 \
                if global_args.optimize_recomp_communication_status > 0 \
                else global_args.optimize_recomp_communication_status

            return output_[:output_.shape[0] // world_size]

    @staticmethod
    def backward(ctx, grad_output):
        return RowSeqParallelLinear.backward(ctx, grad_output)
