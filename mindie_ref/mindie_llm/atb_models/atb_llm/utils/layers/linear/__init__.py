# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
from typing import List

import torch
from atb_llm.utils.log import logger
from torch import nn

from .fast_linear import FastLinear
from ...quantize.smooth_quant.quant_linear import SmoothQuantLinearStatic
from ...quantize.w8a16 import W8A16LinearStatic
from ...quantize.w8a8 import W8A8LinearStatic


def get_linear(weight, bias, quantize, is_norm=False):
    if quantize is None:
        linear = FastLinear(weight, bias, is_norm)
    elif quantize == "smooth_quant":
        try:
            qweight, weight_scales, weight_zeros, act_scales, act_zeros = weight
        except Exception as err:
            logger.error(
                f"The passed weight is not `smooth_quant` compatible, loader needs to be updated."
            )
            raise AssertionError from err
        linear = SmoothQuantLinearStatic(
            weight=qweight,
            weight_scales=weight_scales,
            weight_zeros=weight_zeros,
            act_scales=act_scales,
            act_zeros=act_zeros
        )
    elif quantize == "w8a8":
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, deq_scale, quant_bias, input_scale, input_offset = weight
            except Exception as err:
                logger.error(
                    f"The passed weight is not `w8a8` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A8LinearStatic(
                weight=qweight,
                deq_scale=deq_scale,
                input_scale=input_scale,
                quant_bias=quant_bias,
                input_offset=input_offset
            )
    elif quantize == "w8a16":
        qweight, weight_scale, weight_offset = weight
        linear = W8A16LinearStatic(
            weight=qweight,
            weight_scale=weight_scale,
            weight_offset=weight_offset,
        )
    else:
        logger.error(f"Quantization `{quantize}` is not implemented yet.")
        raise AssertionError
    return linear


class SuperLayer(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, input):
        return self.linear.forward(input)


class TensorHead(SuperLayer):
    def __init__(self, linear):
        super().__init__(linear)

    @staticmethod
    def load_weight(config, prefix: str, weights, is_norm=False):
        weight = weights.get_whole_tensor(f"{prefix}.weight", dim=0)

        # GPTQ doesn't quantize heads (nor embeddings)
        if config.quantize == "gptq":
            quantize = None
        else:
            quantize = config.quantize
        return TensorHead(
            get_linear(weight, bias=None, quantize=quantize, is_norm=is_norm),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        out = torch.mm(input, self.linear.weight.T)
        return out


class TensorParallelHead(SuperLayer):
    def __init__(self, linear, process_group, should_gather: bool):
        super().__init__(linear)
        self.process_group = process_group
        self.should_gather = should_gather

    @staticmethod
    def load_weight(config, prefix: str, weights, is_norm=False):
        weight = weights.get_tensor(f"{prefix}.weight")
        should_gather = False

        # GPTQ doesn't quantize heads (nor embeddings)
        if config.quantize == "gptq":
            quantize = None
        else:
            quantize = config.quantize
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=quantize, is_norm=is_norm),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    @staticmethod
    def load(config, prefix: str, weights, is_norm=False):
        if weights.process_group.size() > 1:
            try:
                weight = weights.get_sharded(f"{prefix}.weight", dim=0)
                should_gather = True
            except AssertionError:
                # If the vocab size is not divisible by number of shards
                # just load the entire thing.
                weight = weights.get_tensor(f"{prefix}.weight")
                should_gather = False
        else:
            weight = weights.get_tensor(f"{prefix}.weight")
            should_gather = False

        # GPTQ doesn't quantize heads (nor embeddings)
        if config.quantize == "gptq":
            quantize = None
        else:
            quantize = config.quantize
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=quantize, is_norm=is_norm),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.should_gather:
            return super().forward(input)

        world_size = self.process_group.size()
        if len(input.shape) == 2 and isinstance(self.linear, FastLinear):
            out_dim = self.linear.weight.shape[0]
            if input.shape[0] == 1:
                world_out = input.new_empty(1, out_dim * world_size)
                local_out = input.new_empty(1, out_dim)
                gather_input = local_out
            else:
                world_out = input.new_empty(out_dim * world_size, input.shape[0])
                gather_input = input.new_empty(out_dim, input.shape[0])
                local_out = gather_input.T

            torch.mm(input, self.linear.weight.T, out=local_out)
            torch.distributed.all_gather_into_tensor(
                world_out, gather_input, group=self.process_group
            )

            if input.shape[0] == 1:
                return world_out
            return world_out.T

        output = super().forward(input)
        world_output = [
            torch.empty_like(output)
            for _ in range(self.process_group.size())
        ]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        return world_output


class TensorParallelColumnLinear(SuperLayer):
    @classmethod
    def load_qkv(cls, config, prefix: str, weights, bias: bool, num_heads: int, num_kv_heads: int = None):
        """Specific method when the QKV was joined after the fact"""
        if num_kv_heads is None:
            num_kv_heads = num_heads
        weight = weights.get_weights_col_packed_qkv(
            prefix, quantize=config.quantize, num_heads=num_heads, num_kv_heads=num_kv_heads
        )
        if bias:
            bias = weights.get_tensor_col_packed_qkv(
                f"{prefix}.bias", num_heads=num_heads, num_kv_heads=num_kv_heads
            )
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)

    @classmethod
    def load_gate_up(cls, config, prefix: str, weights, bias: bool):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_mlp(
            prefix, quantize=config.quantize
        )
        if bias:
            bias = weights.get_tensor_col_packed_mlp(f"{prefix}.bias")
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        return cls.load_multi(config, [prefix], weights, bias, dim=0)

    @classmethod
    def load_multi(cls, config, prefixes: List[str], weights, bias: bool, dim: int):
        weight = weights.get_multi_weights_col(
            prefixes, quantize=config.quantize, dim=dim
        )

        if bias:
            b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
            bias = torch.cat(b, dim=dim)
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)


class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group):
        super().__init__(linear)
        self.process_group = process_group

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

        if bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(
            get_linear(weight, bias, config.quantize),
            process_group=weights.process_group,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out
