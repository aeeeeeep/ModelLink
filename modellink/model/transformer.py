# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
"""Transformer."""
import torch
import torch.nn.functional as F

from ascendspeed.components.moe.moe import MoE
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_expert_parallel_group

from .module import MegatronModule


class MixtralParallelMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.ffn_hidden_size
        self.hidden_dim = config.hidden_size

        self.w1 = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

        self.w2 = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_bias_add=True,
            input_is_parallel=True,
            is_expert=False,
        )

        self.w3 = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

        self.act_fn = F.silu

    def forward(self, hidden_states):
        selects, h = hidden_states.shape
        hidden_states = hidden_states.view(selects, 1, h)
        current_hidden_states = self.act_fn(self.w1(hidden_states)[0]) * self.w3(hidden_states)[0]
        current_hidden_states = self.w2(current_hidden_states)[0].view(selects, h)
        return current_hidden_states


class MixtralParallelMLPBM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.ffn_hidden_size
        self.hidden_dim = config.hidden_size

        self.w1 = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

        self.w2 = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_bias_add=True,
            input_is_parallel=True,
            is_expert=False,
        )

        self.w3 = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

        self.act_fn = F.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)[0]) * self.w3(hidden_states)[0]
        current_hidden_states = self.w2(current_hidden_states)[0]
        return current_hidden_states


class MixtralSparseMoeBlock(MegatronModule):
    """
    This is a megatron implementation refer to HuggingFace Mixtral Model.
    Which strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        args = get_args()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.ffn_hidden_size
        self.num_experts = getattr(args, "num_experts", 8)
        self.top_k = getattr(args, "moe_router_topk", 2)

        # gating
        self.gate = torch.nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = torch.nn.ModuleList(
            [MixtralParallelMLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor):
        s, b, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        # route: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        output_total = torch.zeros_like(hidden_states)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be solicited
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            row, column = torch.where(expert_mask[expert_idx])

            if column.shape[0] == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_hidden_states = expert_layer(hidden_states[column]) * routing_weights[column, row, None]
            output_total[column] = output_total[column] + current_hidden_states.to(hidden_states.dtype)

        output_total = output_total.view(s, b, h).contiguous()

        return output_total, None


class MixtralDenseMoeBlockWithAuxLoss(MegatronModule):
    """
    This is a Mixtral Dense Moe implementation in order to achieve better performance on Static Graph Pattern.
    (To avoid the dynamic shape issue in sparse MOE (Mixture of Experts) model)
    """

    def __init__(self, config):
        super().__init__()
        from megatron.core.transformer.moe.router import TopKRouter

        args = get_args()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.ffn_hidden_size
        self.num_experts = getattr(args, "num_experts", 8)
        self.top_k = getattr(args, "moe_router_topk", 2)
        config.moe_aux_loss_coeff = 1e-2

        self.gate = TopKRouter(
            self.num_experts,
            list(range(self.num_experts)),
            config=config
        )

        self.experts = torch.nn.ModuleList(
            [MixtralParallelMLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor):
        s, b, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        routing_weights, selected_experts = self.gate(hidden_states)

        output_total = torch.zeros_like(hidden_states)
        output_bias = None

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 0, 1)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_hidden_states = expert_layer(hidden_states)

            gather = (expert_mask[expert_idx, ...] * routing_weights).sum(1).unsqueeze(-1)
            output_total = output_total + (expert_hidden_states * gather).to(hidden_states.dtype)

        output_total = output_total.view(s, b, h).contiguous()

        return output_total, output_bias


class MixtralDenseMoeBlock(MegatronModule):
    """
    This is a Mixtral Dense Moe implementation in order to achieve better performance on Static Graph Pattern.
    (To avoid the dynamic shape issue in sparse MOE (Mixture of Experts) model)
    """

    def __init__(self, config):
        super().__init__()
        args = get_args()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.ffn_hidden_size
        self.num_experts = getattr(args, "num_experts", 8)
        self.top_k = getattr(args, "moe_router_topk", 2)

        # gating
        self.gate = torch.nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = torch.nn.ModuleList(
            [MixtralParallelMLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor):
        s, b, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        output_total = torch.zeros_like(hidden_states)
        output_bias = None

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 0, 1)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_hidden_states = expert_layer(hidden_states)

            gather = (expert_mask[expert_idx, ...] * routing_weights).sum(1).unsqueeze(-1)
            output_total = output_total + (expert_hidden_states * gather).to(hidden_states.dtype)

        output_total = output_total.view(s, b, h).contiguous()

        return output_total, output_bias


class GeneralMoeBlock(MegatronModule):
    def __init__(self, config, layer_number=None):
        from megatron.model.transformer import ParallelMLP

        super().__init__()
        args = get_args()

        try:
            expert_parallel_group = get_expert_parallel_group()
        except AssertionError:
            expert_parallel_group = None

        if layer_number is None:
            self.block = MoE(
                args.hidden_size,
                MixtralParallelMLPBM(config, ),
                num_experts=args.num_experts,
                ep_size=args.expert_model_parallel_size,
                k=args.moe_router_topk,
                capacity_factor=args.moe_train_capacity_factor,
                eval_capacity_factor=args.moe_train_capacity_factor,
                aux_loss_coef=args.moe_aux_loss_coeff,
                ep_group=expert_parallel_group,
                noisy_gate_policy=args.noisy_gate_policy
            )
        else:
            if layer_number % args.expert_interval == 0:
                self.block = MoE(
                    args.hidden_size,
                    MixtralParallelMLPBM(config, ),
                    num_experts=args.num_experts,
                    ep_size=args.expert_model_parallel_size,
                    k=args.moe_router_topk,
                    capacity_factor=args.moe_train_capacity_factor,
                    eval_capacity_factor=args.moe_train_capacity_factor,
                    aux_loss_coef=args.moe_aux_loss_coeff,
                    ep_group=expert_parallel_group,
                    noisy_gate_policy=args.noisy_gate_policy
                )
            else:
                self.block = ParallelMLP(config)

    def forward(self, hidden_states, used_token=None):
        output = self.block(hidden_states, used_token)
        return output[0], None
