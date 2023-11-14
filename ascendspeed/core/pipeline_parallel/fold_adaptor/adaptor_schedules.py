# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Iterator, List, Union

import torch
import torch_npu
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ascendspeed.model.distributed import DistributedDataParallel as LocalDDP
import ascendspeed
from ascendspeed import get_args
from ascendspeed.core import parallel_state
from ascendspeed.core.enums import ModelType
from ascendspeed.core.pipeline_parallel.schedules import clear_npu_overflow_flag, \
    deallocate_output_tensor, custom_backward, forward_backward_pipelining_with_interleaving, forward_backward_no_pipelining, forward_backward_pipelining_without_interleaving, optimized_forward_backward_pipelining
from ascendspeed.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
from ascendspeed.global_vars import get_num_microbatches
from ascendspeed.model.lora_utils import get_lora_model_classes, is_enable_lora
from ascendspeed.utils import unwrap_model
from ascendspeed.model import DistributedDataParallel as LocalDDP
from ascendspeed.model import Float16Module
from ascendspeed.error_utils import check_equal, check_type, ensure_var_is_none, ensure_var_is_not_none
from . import adaptor_p2p_comm as p2p_comminication
# Types
Shape = Union[List[int], torch.Size]


def forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data=False,
        checkpoint_activations_microbatch=None,
):
    """
    Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor.
    """
    args = get_args()
    if config.timers is not None and args.foldx_mode is None:
        config.timers('forward-compute', log_level=2).start()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True
    unwrap_model_classes = (torchDDP, LocalDDP, Float16Module)
    if is_enable_lora():
        unwrap_model_classes += get_lora_model_classes()
    unwrapped_model = unwrap_model(model, unwrap_model_classes)
    set_input_tensor = get_attr_wrapped_model(unwrapped_model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            if not args.no_pipeline_parallel:
                output_tensor = loss / num_microbatches
            else:
                output_tensor = loss
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None and args.foldx_mode is None:
        config.timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model=None):
    """
    Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage).
    """

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    args = get_args()
    if args.deepspeed:
        ensure_var_is_not_none(model)

    if config.timers is not None and args.foldx_mode is None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]
    clear_npu_overflow_flag()
    # Backward pass.
    if args.deepspeed:
        model.backward(output_tensor[0])
    else:
        if output_tensor_grad[0] is None and config.grad_scale_func is not None:
            output_tensor[0] = config.grad_scale_func(output_tensor[0])

        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)
    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
            parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None and args.foldx_mode is None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def get_forward_backward_func():
    """
    Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    """

    args = get_args()
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            if args.foldx_mode == "fifo":
                forward_backward_func = forward_backward_pipelining_with_foldx_fifo
            if args.foldx_mode == "aiao":
                forward_backward_func = forward_backward_pipelining_with_foldx_aiao
            check_equal(get_num_microbatches() % args.pipeline_model_parallel_size, 0,
                        error_info='{} not equal {}: '
                                   'number of microbatches is not divisible by pipeline-parallel ' \
                                   'size when using interleaved schedule')
        elif args.optimized_pipeline:
            forward_backward_func = optimized_forward_backward_pipelining
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def forward_backward_pipelining_with_foldx_aiao(*,
                                                forward_step_func,
                                                data_iterator: Union[Iterator, List[Iterator]],
                                                model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                num_microbatches: int,
                                                seq_length: int,  # unused
                                                micro_batch_size: int,  # unused
                                                decoder_seq_length: int = None,  # unused
                                                forward_only: bool = False,
                                                collect_non_loss_data: bool = False, ):
    """Returns dictionary with losses if the last stage, empty dict otherwise."""
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    output_tensor_grads = [[] for _ in range(len(model))]
    config = get_model_config(model[0])
    model_type = get_model_type(model[0])
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks

    num_chunk_warmup_microbatches = get_num_microbatches()
    num_warmup_microbatches = num_microbatches
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (num_chunk_warmup_microbatches * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // num_chunk_warmup_microbatches
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_first_stage() and \
                len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
            input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func=forward_step_func,
                                     data_iterator=data_iterator[model_chunk_id],
                                     model=model[model_chunk_id],
                                     num_microbatches=get_num_microbatches(),
                                     input_tensor=input_tensor, forward_data_store=losses_reduced, config=config)
        output_tensors[model_chunk_id].append(output_tensor)

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model=None)

        return input_tensor_grad

    def init_recv_prev(k):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if k - (pipeline_parallel_size - 1) < 0:
                return False
            next_forward_model_chunk_id = get_model_chunk_id(
                k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                return False
        return True

    def init_recv_next(k):
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            if k - (pipeline_parallel_size - 1) < 0:
                return False
            next_backward_model_chunk_id = get_model_chunk_id(
                k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                return False
        return True

    input_tensors_ops = []
    output_tensor_grads_ops = []

    def gather_input_tensor(k):
        if not (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and
                get_model_chunk_id(k, forward=True) == 0):
            input_tensor, op = input_tensors_ops.pop(0)
            op.wait()
            input_tensors[get_model_chunk_id(k, forward=True)].append(
                p2p_communication.recv_gather(input_tensor))

    def gather_output_tensor_grad(k):
        if not (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and
                get_model_chunk_id(k, forward=False) == (num_model_chunks - 1)):
            output_tensor_grad, op = output_tensor_grads_ops.pop(0)
            op.wait()
            output_tensor_grads[get_model_chunk_id(
                k, forward=False)].append(p2p_communication.recv_gather(output_tensor_grad))

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    if not parallel_state.is_pipeline_first_stage():
        input_tensor, _, ops = p2p_communication.async_communicate(None, None, True, False)
        input_tensors_ops.append((input_tensor, ops[0]))
    for k in range(num_warmup_microbatches):
        # Determine if tensor should be received from previous stage.
        recv_prev = False if k == (num_microbatches - 1) else init_recv_prev(k)

        gather_input_tensor(k)

        if recv_prev:
            input_tensor, _, ops = p2p_communication.async_communicate(None, None, True, False)
            input_tensors_ops.append((input_tensor, ops[0]))

        output_tensor = forward_step_helper(k)
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None
        p2p_communication.async_communicate(output_tensor, None, False, False)

    model_gradient_reduces = []
    if not parallel_state.is_pipeline_last_stage():
        _, output_tensor_grad, ops = p2p_communication.async_communicate(None, None, False, True)
        output_tensor_grads_ops.append((output_tensor_grad, ops[0]))
    for k in range(num_microbatches_remaining, num_microbatches):
        recv_next = init_recv_next(k)
        if k == (num_microbatches - 1):
            recv_next = False

        gather_output_tensor_grad(k)

        if get_model_chunk_id(k, forward=False) < num_model_chunks - 1 and \
                get_model_chunk_id(k, forward=False) < get_model_chunk_id(k - 1, forward=False):
            handles = model[get_model_chunk_id(k, forward=False) + 1].allreduce_gradients(async_op=True)
            model_gradient_reduces.append(handles)
        if recv_next:
            _, output_tensor_grad, ops = p2p_communication.async_communicate(None, None, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))

        input_tensor_grad = backward_step_helper(k)
        p2p_communication.async_communicate(None, input_tensor_grad, False, False)
    handles = model[0].allreduce_gradients(async_op=True)
    model_gradient_reduces.append(handles)
    for handles in model_gradient_reduces:
        for handle in handles:
            handle.wait()

    return losses_reduced


def forward_backward_pipelining_with_foldx_fifo(
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,  # unused
        micro_batch_size: int,  # unused
        decoder_seq_length: int = None,  # unused
        forward_only: bool = False,
        collect_non_loss_data: bool = False, ):
    """Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    output_tensor_grads = [[] for _ in range(len(model))]
    config = get_model_config(model[0])
    model_type = get_model_type(model[0])
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    # Run all forward passes and then all backward passes if number of
    # microbatches is just the number of pipeline stages.
    # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
    # all workers, followed by more microbatches after depending on
    # stage ID (more forward passes for earlier stages, later stages can
    # immediately start with 1F1B).
    if get_num_microbatches() == pipeline_parallel_size:
        num_warmup_microbatches = num_microbatches
        all_warmup_microbatches = True
    else:
        num_warmup_microbatches = \
            (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_microbatches += (
            num_model_chunks - 1) * pipeline_parallel_size
        num_warmup_microbatches = min(num_warmup_microbatches,
                                        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func=forward_step_func,
                                     data_iterator=data_iterator[model_chunk_id],
                                     model=model[model_chunk_id],
                                     num_microbatches=get_num_microbatches(),
                                     input_tensor=input_tensor, forward_data_store=losses_reduced, config=config)
        output_tensors[model_chunk_id].append(output_tensor)

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model=None)

        return input_tensor_grad

    def init_recv_prev(k):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if k - (pipeline_parallel_size - 1) < 0:
                return False
            next_forward_model_chunk_id = get_model_chunk_id(
                k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                return False
        return True

    def init_recv_next(k):
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            if k - (pipeline_parallel_size - 1) < 0:
                return False
            next_backward_model_chunk_id = get_model_chunk_id(
                k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                return False
        return True

    input_tensors_ops = []
    output_tensor_grads_ops = []

    def gather_input_tensor(k):
        if not (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and
                get_model_chunk_id(k, forward=True) == 0):
            input_tensor, op = input_tensors_ops.pop(0)
            op.wait()
            input_tensors[get_model_chunk_id(k, forward=True)].append(
                p2p_communication.recv_gather(input_tensor))

    def gather_output_tensor_grad(k):
        if not (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and
                get_model_chunk_id(k, forward=False) == (num_model_chunks - 1)):
            output_tensor_grad, op = output_tensor_grads_ops.pop(0)
            op.wait()
            output_tensor_grads[get_model_chunk_id(k, forward=False)].append(
                p2p_communication.recv_gather(output_tensor_grad))

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    if not parallel_state.is_pipeline_first_stage():
        input_tensor, _, ops = p2p_communication.async_communicate_group(None, None, True, False)
        input_tensors_ops.append((input_tensor, ops[0]))
    for k in range(num_warmup_microbatches):
        gather_input_tensor(k)
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        recv_prev = init_recv_prev(k)
        if k == (num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if output_tensor is not None and recv_prev:
            input_tensor, _, ops = p2p_communication.async_communicate_group(output_tensor, None, True, False)
            input_tensors_ops.append((input_tensor, ops[0]))        
        elif output_tensor is not None:
            p2p_communication.async_communicate_group(output_tensor, None, False, False)
        elif recv_prev:
            input_tensor, _, ops = p2p_communication.async_communicate_group(None, None, True, False)
            input_tensors_ops.append((input_tensor, ops[0]))
        if k == (num_warmup_microbatches - 1) and not all_warmup_microbatches:
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            if recv_next:
                _, output_tensor_grad, ops = p2p_communication.async_communicate_group(None, None, False, True)
                output_tensor_grads_ops.append((output_tensor_grad, ops[0]))

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        gather_input_tensor(forward_k)
        output_tensor = forward_step_helper(forward_k)
        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None
        # Determine if peers are sending, and where in data structure to put
        # received tensors.forward_only
        recv_prev = init_recv_prev(forward_k)
        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False
        if output_tensor is not None and recv_prev:
            input_tensor, _, ops = p2p_communication.async_communicate_group(output_tensor, None, True, False)
            input_tensors_ops.append((input_tensor, ops[0]))        
        elif output_tensor is not None:
            p2p_communication.async_communicate_group(output_tensor, None, False, False)
        elif recv_prev:
            input_tensor, _, ops = p2p_communication.async_communicate_group(None, None, True, False)
            input_tensors_ops.append((input_tensor, ops[0]))

        # Backward pass.
        backward_k = k
        gather_output_tensor_grad(backward_k)
        input_tensor_grad = backward_step_helper(backward_k)
        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        if parallel_state.is_pipeline_first_stage():
            input_tensor_grad = None
        recv_next = init_recv_next(backward_k)
        if input_tensor_grad is not None and recv_next:
            _, output_tensor_grad, ops = p2p_communication.async_communicate_group(None, input_tensor_grad, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))
        elif input_tensor_grad is not None:
            p2p_communication.async_communicate_group(None, input_tensor_grad, False, False)
        elif recv_next:
            _, output_tensor_grad, ops = p2p_communication.async_communicate_group(None, None, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))

    model_gradient_reduces = []
    # Run cooldown backward passes (flush out pipeline).
    if all_warmup_microbatches:
        if not parallel_state.is_pipeline_last_stage():
            _, output_tensor_grad, ops = p2p_communication.async_communicate_group(None, None, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))
    for k in range(num_microbatches_remaining, num_microbatches):
        gather_output_tensor_grad(k)
        if args.foldx_dp and get_model_chunk_id(k, forward=False) < num_model_chunks - 1:
            if get_model_chunk_id(k, forward=False) < get_model_chunk_id(k - 1, forward=False):
                handles = model[get_model_chunk_id(k, forward=False) + 1].allreduce_gradients(async_op=True)
                model_gradient_reduces.append(handles)
        input_tensor_grad = backward_step_helper(k)
        recv_next = init_recv_next(k)
        if k == (num_microbatches - 1):
            recv_next = False
        if input_tensor_grad is not None and recv_next:
            _, output_tensor_grad, ops = p2p_communication.async_communicate_group(None, input_tensor_grad, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))
        elif input_tensor_grad is not None:
            p2p_communication.async_communicate_group(None, input_tensor_grad, False, False)
        elif recv_next:
            _, output_tensor_grad, ops = p2p_communication.async_communicate_group(None, None, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))
    handles = model[0].allreduce_gradients(async_op=True)
    model_gradient_reduces.append(handles)
    for handles in model_gradient_reduces:
        for handle in handles:
            handle.wait()

    return losses_reduced

