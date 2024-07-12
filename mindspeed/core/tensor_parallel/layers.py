from functools import wraps

import torch
import torch.nn.functional as F

from megatron.training import get_args
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region
)
from megatron.core.tensor_parallel.layers import (linear_with_grad_accumulation_and_async_allreduce, 
                                                  linear_with_frozen_weight)


def vocab_parallel_embedding_forward(self, input_):
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
        # Get the embeddings.

    # For higher accumulation accuracy for bf16 on NPU.
    output_parallel = F.embedding(masked_input, self.weight)

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    # Reduce across all the model parallel GPUs.
    output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output


def row_parallel_nocomm_optimizer_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.optimize_recomp_communication_level == 0:
            output = forward_func(*args, **kwargs)
        else:
            output = row_parallel_nocomm(*args, **kwargs)
        return output
    return row_parallel_forward


def row_parallel_nocomm(self, input_):
    """Forward of RowParallelLinear w/o comm

            Args:
                input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            Returns:
                - output
                - bias
    """
    # Set up backprop all-reduce.
    if self.input_is_parallel:
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if not self.weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=self.weight,
        bias=None,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=False,
        sequence_parallel=False,
    )

    global_args = get_args()
    output_ = output_parallel
    if global_args.optimize_recomp_communication_status < 2:
        if self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        elif self.explicit_expert_comm:  # non-expert only tensor-parallelism
            assert self.skip_bias_add
            output_ = output_parallel
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    global_args.optimize_recomp_communication_status = global_args.optimize_recomp_communication_status + 1 \
        if global_args.optimize_recomp_communication_status > 0 \
        else global_args.optimize_recomp_communication_status
    if not self.skip_bias_add:
        output = (output_ + self.bias) if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias
