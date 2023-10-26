import math

import torch

from .base import PositionEmbedding, PositionEmbeddingConfig
    

class SinePositionalEmbedding(PositionEmbedding):
    r""" sine position embedding """
    def __init__(self, dim_model: int, *args, **kwargs):
        super().__init__()
        self.dim_model = dim_model
        if dim_model <= 0:
            raise Exception("dim_model should be greater than 0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        pos = (
            torch.arange(0, seq_len, device=x.device, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(0, self.dim_model, device=x.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        return output + pos.unsqueeze(0)