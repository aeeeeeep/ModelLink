import torch
import torch_npu
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)


def get_communication_output(input_, reduce_tensor=False):
    tp_world_size = get_tensor_model_parallel_world_size()
    if tp_world_size == 1:
        return input_
    dim_size = list(input_.size())
    if reduce_tensor:
        assert (
                dim_size[0] % tp_world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // tp_world_size
    else:
        dim_size[0] = dim_size[0] * tp_world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    return output


def _gather_along_first_dim_async(input_):
    """Gather tensors and concatenate along the first dimension async."""
    output = get_communication_output(input_)
    handle = torch.distributed._all_gather_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group(), async_op=True
    )
    return output, handle


def _reduce_scatter_along_first_dim_async(input_):
    """Reduce-scatter the input tensor across model parallel group async."""
    output = get_communication_output(input_, reduce_tensor=True)
    handle = torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group(), async_op=True
    )
    return output, handle


class _FusedColumnSeqParallelLoRAFunction(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_, weight, weight_a, weight_b, scaling):
        # 1. gx = gather(x)
        #       a_scale = a * scaling
        #       ax = a_scale * x
        # 2. gax = gather(ax)
        #       output = w * gx
        # 3. bx = b * gax
        # 4. output += bx
        total_input, handle = _gather_along_first_dim_async(input_)
        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        handle.wait()
        total_ax, handle = _gather_along_first_dim_async(ax)
        output = torch.matmul(total_input, weight.t())
        handle.wait()
        bx = torch.matmul(total_ax, weight_b.t())
        output += bx
        ctx.save_for_backward(input_, ax, weight, weight_b)
        ctx.scaling = scaling
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_b, weight, weight_b = ctx.saved_tensors
        total_a, handle = _gather_along_first_dim_async(input_b)
        grad_output_ = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                        grad_output.shape[2])
        grad_gax = grad_output_.matmul(weight_b)
        handle.wait()
        grad_ax, handle = _reduce_scatter_along_first_dim_async(grad_gax)
        grad_input = grad_output.matmul(weight)
        handle.wait()
        grad_sub_input, handle = _reduce_scatter_along_first_dim_async(grad_input)
        x_ = input_.view(input_.shape[0] * input_.shape[1], input_.shape[2])
        grad_weight_b = grad_output_.t().matmul(
            total_a.view(total_a.shape[0] * total_a.shape[1], total_a.shape[2]))
        grad_weight_a = grad_ax.t().matmul(x_) * ctx.scaling
        handle.wait()
        return grad_sub_input, None, grad_weight_a, grad_weight_b, None


class _FusedRowSeqParallelLoRAFunction(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, weight, weight_a, weight_b, scaling):
        # 1. a_scale = a * scaling
        # 2. ax = a_scale * x
        # 3. rax = reduce_scatter(ax)
        #       output = w * x
        # 4. output = reduce_scatter(output)
        #       bx = b * rax
        # 5. output += bx

        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        rax, handle = _reduce_scatter_along_first_dim_async(ax)
        output = torch.matmul(input_, weight.t())
        handle.wait()
        output_parallel, handle = _reduce_scatter_along_first_dim_async(output)
        bx = torch.matmul(rax, weight_b.t())
        group = get_tensor_model_parallel_group()
        rank = torch.distributed.get_rank(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )

        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        world_size = get_tensor_model_parallel_world_size()

        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size
        handle.wait()
        output_parallel += bx
        ctx.save_for_backward(input_, rax, weight, weight_b)
        ctx.scaling = scaling
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        # grad_weight_b = grad_out * scaling * reduce_scatter(a * x)
        #               = grad_out * (scaling * reduce_scatter(a * x))
        #               = grad_out * input_b
        # grad_weight_a = gather(grad_out * scaling * b) * x
        #               = gather(grad_out) * b * x * scaling
        # grad_input = gather(grad_out) * w

        input_, input_b, weight, weight_b = ctx.saved_tensors

        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        grad_input, grad_total_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight, ctx.hcomm_info, ctx.world_size, bias=None, gather_index=0, gather_output=True
        )
        input_b = input_b.view(input_b.shape[0] * input_b.shape[1], input_b.shape[2])
        grad_weight_b = grad_output_.t().matmul(input_b)
        x = input_.view(input_.shape[0] * input_.shape[1], input_.shape[2])
        grad_ax = grad_total_output.matmul(weight_b)
        grad_weight_a = grad_ax.t().matmul(x) * ctx.scaling
        grad_input = grad_input.view_as(input_)
        return grad_input, None, grad_weight_a, grad_weight_b, None


def column_cc_lora_parallel_linear_forward(input_, weight, weight_a, weight_b, scaling, skip_bias_add, bias):
    """Forward of ColumnParallelLinear with Lora

    Args:
        input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        weight (optional): weight tensor to use, compulsory when
            skip_weight_param_allocation is True.

    Returns:
        - output
        - bias

    """
    bias = bias if not skip_bias_add else None
    output_parallel = _FusedColumnSeqParallelLoRAFunction.apply(input_, weight, weight_a, weight_b, scaling)
    if bias is not None:
        output_parallel = output_parallel + bias
    output = output_parallel
    output_bias = bias if skip_bias_add else None
    return output, output_bias


def row_cc_lora_parallel_linear_forward(input_, weight, weight_a, weight_b, scaling, skip_bias_add, bias):
    output_ = _FusedRowSeqParallelLoRAFunction.apply(input_, weight, weight_a, weight_b, scaling)
    if not skip_bias_add:
        output = (output_ + bias) if bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = bias
    return output, output_bias


def CCLoraParallelLinearForward(self, x, *args, **kwargs):
    previous_dtype = x.dtype
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result, bias = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result, bias = self.base_layer(x, *args, **kwargs)
    else:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            weight = self.base_layer.weight
            skip_bias_add = self.base_layer.skip_bias_add
            bias = self.base_layer.bias
            if self.is_paralle_a:
                result, bias = row_cc_lora_parallel_linear_forward(x, weight, lora_A.weight, lora_B.weight, scaling,
                                                                   skip_bias_add, bias)
            else:
                result, bias = column_cc_lora_parallel_linear_forward(x, weight, lora_A.weight, lora_B.weight, scaling,
                                                                      skip_bias_add, bias)
        result = result.to(previous_dtype)
    return result, bias
