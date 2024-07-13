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

import torch

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.training import get_args


def topk_router_routing(self, logits: torch.Tensor):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
    """
    logits = logits.view(-1, self.config.num_moe_experts)
    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    if (
        self.config.tensor_model_parallel_size > 1
        and self.config.moe_token_dispatcher_type == "alltoall"
    ):
        # Gather the logits from the TP region
        logits = gather_from_sequence_parallel_region(logits)

    if self.routing_type == "sinkhorn":
        scores, indices = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "aux_loss":
        scores, indices = self.aux_loss_load_balancing(logits)
    # add softmax_topk for softmax before topk that difference form routing_type is none
    elif self.routing_type == "softmax_topk":
        logits_ = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        scores, indices = torch.topk(logits_, k=self.topk, dim=1)
    elif self.routing_type == "none":
        top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
    else:
        raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

    return scores, indices


def topk_router_forward(self, input: torch.Tensor):
    """
    Forward pass of the router.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: scores and indices.
    """
    args = get_args()
    self.hidden = input.shape[-1]

    # add input_jitter to distinguish whether to use
    if args.input_jitter:
        input = self.apply_input_jitter(input)
    logits = self.gating(input)
    logits = logits.view(-1, self.config.num_moe_experts)

    scores, indices = self.routing(logits)

    return scores, indices
