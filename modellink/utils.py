# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""General utilities."""
import os
import stat
import sys
import enum
import copy
from contextlib import contextmanager

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP
import torch_npu

if str.startswith(torch.cuda.get_device_name(), 'NVIDIA'):
    from apex.multi_tensor_apply import multi_tensor_applier
    import amp_C
    
from megatron.checkpointing import get_checkpoint_tracker_filename, read_metadata, find_checkpoint_rank_0, get_checkpoint_name
from megatron.global_vars import get_args
from megatron.global_vars import get_adlr_autoresume
from megatron.core import parallel_state
from modellink.model.module import param_is_not_shared
from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate
from megatron import get_num_microbatches
from modellink.error_utils import check_type


WRITE_FILE_DEFAULT_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_FILE_DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
                is_aml() and torch.distributed.get_rank() % torch.cuda.device_count() == 0
        ):
            return True
        else:
            return False
    else:
        return True


def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.IntTensor([0]).cuda()
    
    if str.startswith(torch.cuda.get_device_name(), 'NVIDIA'):
        norm, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm,
            dummy_overflow_buf,
            [params_data],
            False # no per-parameter norm
        )
    else :
        norm = torch.norm(params_data, p=2.0)
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=parallel_state.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=parallel_state.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=parallel_state.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if parallel_state.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, lr_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from modellink.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    prefix_indices=None,
                                    loss_on_targets_only=False):
    """
    Build masks and position id for left to right model.
    :param prefix_indices: argument can have multiple types:
        - None signifies that the model is fully autoregressive.
        - List[int] the argument holds all prefix indices that split a row into an input and a target
        - List[List[int]] the argument holds all prefix indices that split documents between input and target.
    :param loss_on_targets_only: bool to determine if we should mask loss on prefix.
    """

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask or prefix_indices is not None:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask or prefix_indices is not None:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]

            # If the last eod token is not the last token of the sequence, we suppose that there is a partial document
            # We treat this case as if we add an eod token at the end of the sequence.
            if data[b][-1] != eod_token:
                eod_index = torch.cat(
                    (eod_index, torch.tensor([len(data[b])], dtype=eod_index.dtype, device=eod_index.device))
                )

            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]

                if reset_attention_mask:
                    # Prevent cross document interactions.
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0

                    # Prefix lm per document.
                    if prefix_indices:
                        check_type(prefix_indices[b], list, error_message=f"prefix for a row has to be document specific, " \
                                                            "and consequently return a list, got {prefix_indices[b]}")
                        attention_mask[b, 0, prev_index: prefix_indices[b][j], prev_index: prefix_indices[b][j]] = 1
                        if loss_on_targets_only:
                            # Last token of the prefix should predict the prefix_index id
                            loss_mask[b, prev_index: prefix_indices[b][j] - 1] = 0.0

                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)

                prev_index = i + 1

            # Prefix lm per row.
            if prefix_indices is not None and (reset_attention_mask is False):
                check_type(prefix_indices[b], int, error_message=f"prefix for a row has to be row specific," \
                                                    " and consequently return an int, got {prefix_indices[b]}")
                attention_mask[b, 0, :prefix_indices[b], :prefix_indices[b]] = 1
                if loss_on_targets_only:
                    # Last token of the prefix should predict the prefix_index id
                    loss_mask[b, :prefix_indices[b] - 1] = 0.0

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_parameters_in_billions(model):
    gpus_per_model = torch.distributed.get_world_size(group=parallel_state.get_model_parallel_group())

    approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p, 'ds_id') else p.nelement() for p in model_module.parameters()])
                                        for model_module in model])

    return approx_parameters_in_billions * gpus_per_model / (1e9)


def throughput_calculator(model, args, iteration_time, total_iterations):
    gpus_per_model = torch.distributed.get_world_size(group=parallel_state.get_model_parallel_group())
    batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
    samples_per_model = batch_size * args.seq_length
    model_replica_count = torch.distributed.get_world_size() / gpus_per_model
    approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
    elapsed_time_per_iter = iteration_time / total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    #flops calculator
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    vocab_size = args.padded_vocab_size

    # General TFLOPs formula.
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 4 if args.checkpoint_activations else 3
    seq_len = args.seq_length
    if hasattr(args, 'actual_seq_length'):
        seq_len = args.actual_seq_length
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
    return samples_per_second, tflops, approx_parameters_in_billions


def checkpoint_throughput_calculator(model, latency_second):
    approx_parameters_in_billions = get_parameters_in_billions(model)
    checkpoint_multiplier = 14  # fp16 weights (2), fp32 weights (4), fp32 momentum (4), fp32 variance (4)
    checkpoint_GB = approx_parameters_in_billions * checkpoint_multiplier
    GB_per_second = checkpoint_GB / latency_second
    print_rank_0(f"Checkpoint Save GB: {round(checkpoint_GB, 3)}, GB/Sec: {round(GB_per_second,2)}, Latency(second): {round(latency_second, 3)}")


def get_tune_attention_mask(attention_mask_1d, reset_attention_mask=True):
    args = get_args()
    micro_batch_size, seq_length = attention_mask_1d.size()
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=attention_mask_1d.device)).view(
        att_mask_batch, 1, seq_length, seq_length)
    attention_mask = attention_mask.masked_fill((attention_mask_1d < 0.5).view(-1, 1, 1, seq_length), value=0)
    attention_mask = (attention_mask < 0.5)
    return attention_mask


def convert_args_to_strs(args):
    args_c = copy.deepcopy(args)
    for name in dir(args_c):
        v = getattr(args_c, name)
        if isinstance(v, enum.Enum):
            setattr(args_c, name, v.name)
    return args_c


def _load_base_checkpoint(load_dir, rank0=False):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                         'random')
        return None, "", False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
        if release:
            print_rank_0(f' loading release checkpoint from {load_dir}')
        else:
            print_rank_0(f' loading checkpoint from {load_dir} at iteration {iteration}')

    # Load the checkpoint.
    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)

    return state_dict, checkpoint_name, release


def load_args_from_checkpoint(args, load_arg='load'):
    """Set required arguments from the checkpoint specified in the
    arguments.

    Will overwrite arguments that have a non-None default value, but
    will leave any arguments that default to None as set.

    Returns the same args NameSpace with the new values added/updated.

    If no checkpoint is specified in args, or if the checkpoint is
    there but invalid, the arguments will not be modified

    """
    load_dir = getattr(args, load_arg)

    if load_dir is None:
        print_rank_0('No load directory specified, using provided arguments.')
        return args

    state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=True)

    # Args.
    if not state_dict:
        print_rank_0('Checkpoint not found to provide arguments, using provided arguments.')
        return args

    if 'args' not in state_dict:
        print_rank_0('Checkpoint provided does not have arguments saved, using provided arguments.')
        return args

    checkpoint_args = state_dict['args']
    checkpoint_version = state_dict.get('checkpoint_version', 0)
    args.iteration = state_dict['iteration']

    # One-off conversion for foundation models
    if hasattr(checkpoint_args, 'disable_bias_linear'):
        setattr(checkpoint_args, 'add_bias_linear', not getattr(checkpoint_args, 'disable_bias_linear'))

    def _set_arg(arg_name, old_arg_name=None, force=False):
        if not force and getattr(args, arg_name, None) is not None:
            return

        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name, None)

        if checkpoint_value is not None:
            print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
            setattr(args, arg_name, checkpoint_value)
        else:
            print_rank_0(f"Checkpoint did not provide arguments {arg_name}")

    _set_arg('num_layers')
    _set_arg('hidden_size')
    _set_arg('ffn_hidden_size')
    _set_arg('seq_length')
    _set_arg('num_attention_heads')
    _set_arg('num_query_groups', force=True)
    _set_arg('group_query_attention', force=True)
    _set_arg('kv_channels')
    _set_arg('max_position_embeddings')
    _set_arg('position_embedding_type', force=True)
    _set_arg('add_position_embedding', force=True)
    _set_arg('use_rotary_position_embeddings', force=True)
    _set_arg('rotary_percent', force=True)
    _set_arg('add_bias_linear', force=True)
    _set_arg('swiglu', force=True)
    _set_arg('untie_embeddings_and_output_weights', force=True)
    _set_arg('apply_layernorm_1p', force=True)
    _set_arg('normalization', force=True)
    _set_arg('tokenizer_type')
    _set_arg('padded_vocab_size')
    if checkpoint_version < 3.0:
        _set_arg('tensor_model_parallel_size',
                 'model_parallel_size')
    else:
        _set_arg('tensor_model_parallel_size', force=True)
        _set_arg('pipeline_model_parallel_size', force=True)
        _set_arg('virtual_pipeline_model_parallel_size', force=True)
        _set_arg('num_layers_per_virtual_pipeline_stage')
    return args, checkpoint_args


@contextmanager
def cpu_tensor_reduce_context(enable=False):
    def cpu_reduce_ex(self, proto):
        self = self.cpu()
        return torch_npu.utils.storage._reduce_ex(self, proto)

    org_reduce_ex = torch.Tensor.__reduce_ex__
    try:
        if enable:
            torch.Tensor.__reduce_ex__ = cpu_reduce_ex
        yield
    finally:
        torch.Tensor.__reduce_ex__ = org_reduce_ex
