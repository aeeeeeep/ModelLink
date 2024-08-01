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
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, save_to_aux_losses_tracker
from megatron.core import parallel_state

from .moe_utils import topk_softmax_with_capacity, switch_load_balancing_loss_func


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
        # A naive top-k routing without load balancing
        # top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
        # scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        scores, indices, _ = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
        )
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


def aux_loss_load_balancing(self, logits: torch.Tensor):
    """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
    """
    probs, indices, tokens_per_expert = topk_softmax_with_capacity(
        logits,
        self.topk,
        capacity_factor=self.config.moe_expert_capacity_factor,
        pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        drop_policy=self.config.moe_token_drop_policy,
    )

    # Apply load balancing loss
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)
    return probs, indices


def apply_load_balancing_loss(
    self,
    probs: torch.Tensor,
    num_local_tokens_per_expert: torch.Tensor,
    activation: torch.Tensor,
):
    """Applies auxiliary loss to the MoE layer.

    Args:
        probs (torch.Tensor): The probabilities output by the MoE layer.
        num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert.
        activation (torch.Tensor): The activation tensor to attach the gradient function to.

    Returns:
        torch.Tensor: The activation tensor with the attached gradient function.
    """
    moe_aux_loss_coeff = (
            self.config.moe_aux_loss_coeff / parallel_state.get_tensor_model_parallel_world_size()
    )
    aux_loss = switch_load_balancing_loss_func(
        probs, num_local_tokens_per_expert, self.topk, moe_aux_loss_coeff
    )
    save_to_aux_losses_tracker(
        "load_balancing_loss",
        aux_loss / moe_aux_loss_coeff,
        self.layer_number,
        self.config.num_layers,
    )
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)

    return activation

