# coding=utf-8
# Copyright (c) 2020, HUAWEI CORPORATION.  All rights reserved.
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

"""Megatron optimizer."""
import os
from abc import ABC
from abc import abstractmethod

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from ascendspeed import get_timers
from ascendspeed import print_rank_0
from ascendspeed.core import parallel_state, tensor_parallel
from ascendspeed.model.module import param_is_not_shared
from ascendspeed.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate
from ascendspeed.utils import unwrap_model
from ascendspeed.model import Float16Module, DistributedDataParallel as LocalDDP
from deepspeed.accelerator import get_accelerator
from ascendspeed.model.lora_utils import is_enable_lora, get_lora_model_classes
from ascendspeed.error_utils import check_equal, ensure_valid, check_divisible_by_zero
from .clip_grads import clip_grad_norm_fp32, count_zeros_fp32


def clear_silent_check():
    if int(os.getenv('NPU_DETECT', '0')):
        from torch_npu.utils.silent_error import clear_hookmodule_list
        clear_hookmodule_list()


def exec_silent_check(loss_scale):
    if int(os.getenv('NPU_DETECT', '0')):
        from torch_npu.utils.silent_error import silent_fault_check
        silent_fault_check(loss_scale)


def get_silent_check_flag():
    found_silent_flag = False
    if int(os.getenv('NPU_DETECT', '0')):
        from torch_npu.utils.silent_error import get_silent_check
        found_silent_flag = get_silent_check() > 0
    return found_silent_flag


def print_silent_check_log():
    import torch_npu
    if hasattr(torch_npu.npu, "print_error_plog"):
        torch_npu.npu.print_error_plog("NPUCheckEvent:AICore Numerical error happen, skip this step!")


def _zero_grad_group_helper(group, set_to_none):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(this, that, overflow_buf=None):
    """Use multi-tensor-applier to copy values from one list to another.
    We don't have a blfoat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16."""
    if get_accelerator().device_name() == 'cuda' and overflow_buf:
        from apex.multi_tensor_apply import multi_tensor_applier
        import amp_C

        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             overflow_buf,
                             [this, that],
                             1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class MegatronOptimizer(ABC):

    def __init__(self, optimizer, clip_grad,
                 log_num_zeros_in_grad,
                 params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp,
                 models):

        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        ensure_valid(self.optimizer, error_message='no optimizer is provided.')
        # Set gradient clipping and logging params.
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.params_have_main_grad = params_have_main_grad
        self.use_contiguous_buffers_in_local_ddp = use_contiguous_buffers_in_local_ddp

        # 'models' are retained for access to the contiguous grad buffers.
        # (see distributed optimizer)
        self.models = models

        if self.use_contiguous_buffers_in_local_ddp:
            ensure_valid(self.params_have_main_grad, error_message="use of contiguous" \
                                                     " buffer requires that params have main grad")

        self.unwrap_model_classes = (torchDDP, LocalDDP, Float16Module)
        if is_enable_lora():
            self.unwrap_model_classes += get_lora_model_classes()

    def get_parameters(self):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

    def get_main_grads_for_grad_norm(self):

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        params = self.get_parameters()
        grads_for_norm = []
        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        return grads_for_norm

    def get_model_parallel_group(self):
        """Default returned here, but the distributed optimizer overrides this."""
        return parallel_state.get_model_parallel_group()

    def clip_grad_norm(self, clip_grad):
        params = self.get_parameters()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        use_global_grad_norm = getattr(self, "use_global_grad_norm", False)
        return clip_grad_norm_fp32(
            params, grads_for_norm, clip_grad,
            model_parallel_group=self.get_model_parallel_group(),
            use_global_grad_norm=use_global_grad_norm)

    def get_clip_grad_norm(self, clip_grad, use_global_grad_norm=False):
        # Norm parameters.
        max_norm = float(clip_grad)
        norm_type = float(2)
        total_norm = 0.0
        params = self.get_parameters()
        for _, param in enumerate(params):
            grad = param.fp16_grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                inv_scale = self.grad_scaler.inv_scale if self.grad_scaler else self.inv_scale
                fp32_unscale_grad = grad.float().mul_(inv_scale)
                grad_norm = torch.norm(fp32_unscale_grad, norm_type)
                total_norm += grad_norm ** norm_type
        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm,
                                    op=torch.distributed.ReduceOp.SUM,
                                    group=parallel_state.get_model_parallel_group())
        if use_global_grad_norm:
            torch.distributed.all_reduce(total_norm,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=parallel_state.get_data_parallel_group())
       
        total_norm = total_norm.item() ** (check_divisible_by_zero(1.0, norm_type))
        clip_coeff = min(1.0, check_divisible_by_zero(max_norm, total_norm + 1.0e-6))
        return total_norm, clip_coeff
    
    def count_zeros(self):
        params = self.get_parameters()
        return count_zeros_fp32(params,
                                model_parallel_group=self.get_model_parallel_group())

    @abstractmethod
    def zero_grad(self, set_to_none=True):
        pass

    @abstractmethod
    def get_loss_scale(self):
        """The output should be a cuda tensor of size 1."""
        pass

    def scale_loss(self, loss):
        """Simple scaling."""
        return self.get_loss_scale() * loss

    @abstractmethod
    def reload_model_params(self):
        """
        Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated.
        """
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    @abstractmethod
    def step(self, args, timers):
        pass

    def gather_model_params(self, args, timers):
        """
        For the case of a non-distributed-optimizer, there is nothing to
        do here.
        """
        pass

    def allreduce_word_embedding_grads(self, args):
        """
        All-reduce word embedding grads.

        Reduce grads across first and last stages to ensure that word_embeddings
        parameters stay in sync. This should only run for models that support
        pipelined model parallelism (BERT and GPT-2).
        """

        if parallel_state.is_rank_in_embedding_group(ignore_virtual=True) and \
                parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                unwrapped_model = self.models[0]
            elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                unwrapped_model = self.models[-1]
            else:  # We do not support the interleaved schedule for T5 yet.
                unwrapped_model = self.models[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, self.unwrap_model_classes)

            if unwrapped_model.share_embeddings_and_output_weights:
                word_embeddings_weight = unwrapped_model.shared_embedding_or_output_weight()
                if word_embeddings_weight.requires_grad:
                    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
                        grad = word_embeddings_weight.main_grad
                    else:
                        grad = word_embeddings_weight.grad
                    torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())

    def allreduce_position_embedding_grads(self, args):
        """
        All-reduce position_embeddings grad across first (encoder) and
        split (decoder) stages to ensure that position embeddings parameters
        stay in sync. This should only run for T5 models with pipeline
        parallelism.
        """
        if parallel_state.is_rank_in_position_embedding_group() and \
                parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
                args.pipeline_model_parallel_split_rank is not None:
            unwrapped_model = self.models[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, self.unwrap_model_classes)
            error_info = 'T5 model is only supported with local DDP mode'
            check_equal(args.DDP_impl, 'local', error_info)
            grad = unwrapped_model.language_model.embedding.position_embeddings.weight.main_grad
            torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())

    def allreduce_embedding_grads(self, args):
        """All-reduce both word and position embeddings."""
        self.allreduce_word_embedding_grads(args)
        self.allreduce_position_embedding_grads(args)

    def allreduce_layernorm_grads(self, args):
        """All-reduce layernorm grads (for sequence parallelism)."""

        # All-reduce layernorm parameters across model parallel nodes
        # when sequence parallelism is used
        if parallel_state.get_tensor_model_parallel_world_size() > 1 and \
                args.sequence_parallel:
            grads = []
            for model_module in self.models:
                unwrapped_model = unwrap_model(
                    model_module, self.unwrap_model_classes)
                for param in unwrapped_model.parameters():
                    if getattr(param, 'sequence_parallel', False) and param.requires_grad:
                        grad = param.main_grad if (args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp)\
                            else param.grad
                        grads.append(grad.data)

            if not grads:
                return
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group())
            for buf, synced in zip(grads, _unflatten_dense_tensors(
                    coalesced, grads)):
                buf.copy_(synced)

    def reduce_model_grads(self, args, timers):
        """All-reduce all grads, and all-reduce embeddings."""

        # All-reduce layer-norm grads (for sequence parallelism).
        timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.allreduce_layernorm_grads(args)
        timers('layernorm-grads-all-reduce').stop()

        # All-reduce if needed.
        if args.DDP_impl == 'local' and args.foldx_mode is None:
            timers('grads-all-reduce', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            for model in self.models:
                model.allreduce_gradients()
            timers('grads-all-reduce').stop()

        # All-reduce embedding grads.
        timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.allreduce_embedding_grads(args)
        timers('embedding-grads-all-reduce').stop()


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Base class for both the float-16 and the distributed optimizer.

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        use_contiguous_buffers_in_local_ddp: if true, the local DDP model
            is using a contiguous buffer to hold the model grads.
        fp16: if true, the model is running in fp16.
        bf16: if true, the model is running in bfloat16.
        params_dtype: used by distributed optimizer.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        models: list of models (i.e., the virtual pipelining models). This
            is used by the distributed optimizer for mapping parameters.
    """

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler,
                 models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            models)

        self.fp16 = fp16
        self.bf16 = bf16
        self.params_dtype = params_dtype
        self.grad_scaler = grad_scaler

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            ensure_valid(not self.fp16, error_message='fp16 expects a grad scaler.')

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.cuda.FloatTensor([1.0])

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()

        # Reset found inf.
        self.found_inf.fill_(0.0)

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)

        # Update across all model parallel instances.
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=self.get_model_parallel_group())
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=parallel_state.get_data_parallel_group())

        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)

        return found_inf_flag

    @torch.no_grad()
    def step(self, args, timers):

        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            timers('optimizer-unscale-and-check-inf', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                clear_silent_check()
                return False, None, None

        loss_scale = 1.0 if self.grad_scaler is None else self.grad_scaler.inv_scale.item()
        exec_silent_check(loss_scale)

        # Clip the main gradients.
        timers('optimizer-clip-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        found_silent_flag = get_silent_check_flag()
        if not found_silent_flag or not (int(os.getenv('NPU_RECOVERY', '0'))):
            # Count the zeros in the grads.
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            num_zeros_in_grad = self.count_zeros() if \
                self.log_num_zeros_in_grad else None
            timers('optimizer-count-zeros').stop()

            # Step the optimizer.
            timers('optimizer-inner-step', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            self.optimizer.step()
            timers('optimizer-inner-step').stop()

            # Update params from main params.
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            self._copy_main_params_to_model_params()
            timers('optimizer-copy-main-to-model-params').stop()

        else:
            # The silent error is found, and skip the step, then call print_error_plog api to print log in plog.
            print_silent_check_log()
            return False, None, None

        # Successful update.
        return True, grad_norm, num_zeros_in_grad


class Float16OptimizerWithFloat16Params(MegatronOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a contihuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
    """

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler, models):

        super(Float16OptimizerWithFloat16Params, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp, models)

        self.bf16 = bf16
        self.grad_scaler = grad_scaler
        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            ensure_valid(self.bf16, error_message='fp16 expects a grad scaler.')

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = get_accelerator().FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = get_accelerator().IntTensor([0])

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = get_accelerator().FloatTensor([1.0])

        # ======================
        # main parameter stuff
        # ======================

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    param_type = param.type().replace('cuda', get_accelerator().device_name())

                    # float16 params:
                    if param_type in ['torch.{}.HalfTensor'.format(get_accelerator().device_name()),
                                      'torch.{}.BFloat16Tensor'.format(get_accelerator().device_name())]:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param,
                                                                  param)
                        if hasattr(param, 'shared'):
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param
                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] \
                                = self.optimizer.state.pop(param)

                    # fp32 params.
                    elif param_type == 'torch.{}.FloatTensor'.format(format(get_accelerator().device_name())):
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param

                    else:
                        device_name = get_accelerator().device_name()
                        raise TypeError('Wrapped parameters must be one of '
                                        'torch.{}.FloatTensor,  '
                                        'torch.{}.HalfTensor, or '
                                        'torch.{}.BFloat16Tensor. '
                                        'Received {}'.format(device_name, device_name, device_name, param.type()))

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(
                fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def zero_grad(self, set_to_none=True):
        """
        We only need to zero the model related parameters, i.e.,
                float16_groups & fp32_from_fp32_groups.
        """
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                # if self.params_have_main_grad:
                if self.params_have_main_grad:
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()
        # For fp32 grads, we need to reset the grads to main grad.
        if self.params_have_main_grad:
            for model_group in self.fp32_from_fp32_groups:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad

    def _unscale_main_grads_and_check_for_nan(self):
        main_grads = []
        # fp32 params fromm float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)
        # Update across all model parallel instances.
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=parallel_state.get_model_parallel_group())

        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)
        return found_inf_flag

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=main_data, that=model_data,
                                        overflow_buf=self._dummy_overflow_buf)

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=model_data, that=main_data,
                                        overflow_buf=self._dummy_overflow_buf)

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    @torch.no_grad()
    def step(self, args, timers):

        timers = get_timers()

        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad', log_level=1).start()
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            timers('optimizer-unscale-and-check-inf', log_level=1).start()
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                clear_silent_check()
                return False, None, None

        loss_scale = 1.0 if self.grad_scaler is None else self.grad_scaler.inv_scale.item()
        exec_silent_check(loss_scale)

        # Clip the main gradients.
        timers('optimizer-clip-main-grad', log_level=1).start()
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        found_silent_flag = get_silent_check_flag()
        if not found_silent_flag or not (int(os.getenv('NPU_RECOVERY', '0'))):
            # count the zeros in the grads
            num_zeros_in_grad = self.count_zeros() if \
                self.log_num_zeros_in_grad else None

            # Step the optimizer.
            self.optimizer.step()

            # Update params from main params.
            timers('optimizer-copy-main-to-model-params', log_level=1).start()
            self._copy_main_params_to_model_params()
            timers('optimizer-copy-main-to-model-params').stop()
        else:
            # The silent error is found, and skip the step, then call print_error_plog api to print log in plog.
            print_silent_check_log()
            return False, None, None

        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        # Optimizer.
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            print_rank_0('***WARNING*** loading optimizer from '
                         'an old checkpoint ...')
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            print_rank_0('***WARNING*** found an old checkpoint, will not '
                         'load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                print_rank_0('***WARNING*** fould the grad scaler in the '
                             'checkpoint but it is None in the class. '
                             'Skipping loading grad scaler ...')

        # Copy data for the main params.
        fp32_from_float16_params_key = 'fp32_from_fp16_params'
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(
                self.fp32_from_float16_groups,
                state_dict[fp32_from_float16_params_key]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


class Float16OptimizerWithoutFp32Grad(Float16OptimizerWithFloat16Params):
    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler, models):

        super(Float16OptimizerWithoutFp32Grad, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, params_dtype, grad_scaler, models)
        self.inv_scale = get_accelerator().FloatTensor([1.0])

    def _main_grad_check_for_nan(self):
        fp16_grads = []
        fp32_grads = []
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.fp16_grad is not None:
                    fp16_grads.append(main_param.fp16_grad.data)
        self.found_inf.fill_(0.0)
        torch._amp_foreach_non_finite_check_and_unscale_(
            fp16_grads, self.found_inf, self.inv_scale)
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=parallel_state.get_model_parallel_group())
       
        found_inf_flag = (self.found_inf.item() > 0)
        if found_inf_flag:
            return found_inf_flag
       
        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    fp32_grads.append(main_param.grad.data)
        self.found_inf.fill_(0.0)
        torch._amp_foreach_non_finite_check_and_unscale_(
            fp32_grads, self.found_inf, self.inv_scale)
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=parallel_state.get_model_parallel_group())
        return found_inf_flag

    def _copy_model_fp16_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                # if self.params_have_main_grad:
                if self.params_have_main_grad:
                    main_param.fp16_grad = model_param.main_grad
                else:
                    if model_param.grad is not None:
                        main_param.fp16_grad = model_param.grad
        # For fp32 grads, we need to reset the grads to main grad.
        if self.params_have_main_grad:
            for model_group in self.fp32_from_fp32_groups:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad

    @torch.no_grad()
    def step(self, args, timers):
        timers('optimizer-copy-to-main-grad', log_level=1).start()
        self._copy_model_fp16_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()
        if self.grad_scaler:
            timers('optimizer-check-inf-and-nan', log_level=1).start()
            found_inf_flag = self._main_grad_check_for_nan()
            timers('optimizer-check-inf-and-nan').stop()
            self.grad_scaler.update(found_inf_flag)
            if found_inf_flag:
                clear_silent_check()
                return False, None, None
        timers('optimizer-get-clip-grad-norm', log_level=1).start()
        grad_norm = None

        loss_scale = 1.0 if self.grad_scaler is None else self.grad_scaler.inv_scale.item()
        exec_silent_check(loss_scale)

        norm_coeff_scale = self.grad_scaler.inv_scale if self.grad_scaler else self.inv_scale
        if self.clip_grad > 0.0:
            grad_norm, clip_coeff = self.get_clip_grad_norm(self.clip_grad)
            norm_coeff_scale = norm_coeff_scale * clip_coeff

        found_silent_flag = get_silent_check_flag()
        if not found_silent_flag or not (int(os.getenv('NPU_RECOVERY', '0'))):

            timers('optimizer-get-clip-grad-norm').stop()
            self.optimizer.step(norm_coeff_scale=norm_coeff_scale)
            # Update params from main params.
            timers('optimizer-copy-main-to-model-params', log_level=1).start()
            self._copy_main_params_to_model_params()
            timers('optimizer-copy-main-to-model-params').stop()
            # Successful update.
            return True, grad_norm, None
        else:
            # The silent error is found, and skip the step, then call print_error_plog api to print log in plog.
            print_silent_check_log()
            return False, None, None


class FP32Optimizer(MegatronOptimizer):

    def __init__(self, optimizer, clip_grad,
                 log_num_zeros_in_grad,
                 params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp,
                 model):

        super(FP32Optimizer, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            model)

        self._scale = get_accelerator().FloatTensor([1.0])

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    @torch.no_grad()
    def step(self):
        """
        Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow.
        """

        # Copy main_grads to grads.
        if self.params_have_main_grad:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = param.main_grad

        exec_silent_check(1.0)
        # Clip gradients.
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)

        found_silent_flag = get_silent_check_flag()
        if not found_silent_flag or not (int(os.getenv('NPU_RECOVERY', '0'))):
            # count the zeros in the grads
            num_zeros_in_grad = self.count_zeros() if \
                self.log_num_zeros_in_grad else None

            # Update parameters.
            self.optimizer.step()
        else:
            # The silent error is found, and skip the step, then call print_error_plog api to print log in plog.
            print_silent_check_log()
            return False, None, None

        # No overflow for FP32 optimizer.
        return True, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
