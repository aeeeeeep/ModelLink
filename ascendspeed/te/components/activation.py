from enum import Enum
from typing import Optional

import torch
from torch import nn


class Activation(str, Enum):
    SquaredReLU = "squared_relu"
    GeLU = "gelu"
    LeakyReLU = "leaky_relu"
    ReLU = "relu"
    SmeLU = "smelu"
    StarReLU = "star_relu"


# For unit testing / parity comparisons, probably not the fastest way
class SquaredReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = torch.nn.functional.relu(x)
        return x_ * x_


class StarReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = torch.nn.functional.relu(x)
        return 0.8944 * x_ * x_ - 0.4472


class SmeLU(nn.Module):
    def __init__(self, beta: float = 2.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = get_epsilon()
        relu = torch.where(
            x >= self.beta,
            x,
            torch.tensor([0.0], device=x.device, dtype=x.dtype),
        )
        return torch.where(
            torch.abs(x) <= self.beta,
            ((x + self.beta) ** 2).type_as(x) / (4.0 * (self.beta + epsilon)),
            relu,
        )


def build_activation(activation: Optional[Activation]):
    if not activation:
        return nn.Identity()

    activation_dict = {
        Activation.ReLU: nn.ReLU,
        Activation.GeLU: nn.GELU,
        Activation.LeakyReLU: nn.LeakyReLU,
        Activation.SquaredReLU: SquaredReLU,
        Activation.StarReLU: StarReLU,
        Activation.SmeLU: SmeLU,
    }
    return activation_dict.get(activation, nn.ReLU)()


def get_epsilon():
    '''
    return the epsilon for calculation to avoid zero divisor
    '''
    epsilon = 1e-8
    return epsilon