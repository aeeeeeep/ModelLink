# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import types
from copy import deepcopy
from functools import wraps
import torch
from mindspeed.moe.utils import MoEAuxLossAutoScaler

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
from megatron.training import get_args


def moe_layer_init_wrapper(init_func):
    @wraps(init_func)
    def moe_layer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        global_args = get_args()

        if global_args.moe_intermediate_size:
            self.config.ffn_hidden_size = global_args.moe_intermediate_size
        if self.config.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)

        if global_args.n_shared_experts:
            config = deepcopy(self.config)
            config.ffn_hidden_size = global_args.n_shared_experts * self.config.ffn_hidden_size
            self.shared_experts = MLP(config, MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                                                 linear_fc2=RowParallelLinear,))

    return moe_layer_init


def moe_layer_forward(self, hidden_states: torch.Tensor):
    # process MoE
    scores, indices = self.router(hidden_states)
    (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
        hidden_states, scores, indices
    )
    router_expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
    output, mlp_bias = self.token_dispatcher.token_unpermutation(router_expert_output, mlp_bias)
    args = get_args()
    if args.moe_router_load_balancing_type == "group_limited_greedy":
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            self.router.l_aux / args.moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
        )
        output = MoEAuxLossAutoScaler.apply(output, self.router.l_aux)
    if args.n_shared_experts:
        share_experts_output, share_experts_bias = self.shared_experts(hidden_states)
        output = output + share_experts_output
        if self.token_dispatcher.add_bias:
            mlp_bias = mlp_bias + share_experts_bias
    return output, mlp_bias