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
import torch.nn.functional as F

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.training import get_args


def group_limited_greedy_topKgating(self, logits: torch.Tensor):
    args = get_args()

    if self.config.moe_token_dispatcher_type == "alltoall":
        seq_length = args.seq_length
    else:
        seq_length = args.seq_length // self.config.tensor_model_parallel_size

    scores = F.softmax(logits, dim=1)
    group_scores = (
        scores.view(args.micro_batch_size * seq_length, args.n_group, -1).max(dim=-1).values
    )  # [n, n_group]

    group_idx = torch.topk(
        group_scores, k=args.topk_group, dim=-1, sorted=False
    )[
        1
    ]  # [n, top_k_group]

    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(
            args.micro_batch_size * seq_length, args.n_group, args.num_experts // args.n_group
        )
        .reshape(args.micro_batch_size * seq_length, -1)
    )  # [n, e]

    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    topk_weight, topk_idx = torch.topk(
        tmp_scores, k=args.moe_router_topk, dim=-1, sorted=False
    )

    ### norm gate to sum 1
    if args.moe_router_topk > 1 and args.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    else:
        topk_weight = topk_weight * args.routed_scaling_factor
    if self.training and self.config.moe_aux_loss_coeff > 0:
        scores_for_aux = scores
        topk_idx_for_aux_loss = topk_idx.view(args.micro_batch_size, -1)
        # aux_topk = self.top_k
        # always compute aux loss based on the naive greedy topk method
        if args.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(args.micro_batch_size, seq_length, -1)

            ce = torch.zeros(
                args.micro_batch_size, args.num_experts, device=logits.device
            )
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(args.micro_batch_size, seq_length * args.moe_router_topk, device=logits.device),
            ).div_(seq_length * args.moe_router_topk / args.num_experts)
            l_aux = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * self.config.moe_aux_loss_coeff
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=args.num_experts
            )
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * args.num_experts
            l_aux = (Pi * fi).sum() * self.config.moe_aux_loss_coeff
    else:
        l_aux = None
    self.l_aux = l_aux
    return topk_weight, topk_idx

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
    elif self.routing_type == "group_limited_greedy":
        scores, indices = group_limited_greedy_topKgating(self, logits)
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
