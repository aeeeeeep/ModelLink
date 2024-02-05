# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
from torch import nn


class FastLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

        return super(FastLayerNorm, self).forward(hidden_states), residual
