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
"""initialization."""

import random
import os
import time
import numpy as np

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ascendspeed import get_adlr_autoresume
from ascendspeed import get_args
from ascendspeed import get_tensorboard_writer
from ascendspeed import mpu
from ascendspeed.core import parallel_state
from ascendspeed.global_vars import set_global_variables
from ascendspeed.core.parallel_state import (set_tensor_model_parallel_rank,
                                             set_tensor_model_parallel_world_size)

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.engine import DeepSpeedEngine, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.utils import log_dist, logger
from deepspeed.git_version_info import version, git_hash, git_branch
from deepspeed.runtime import zero
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from deepspeed.pipe import PipelineModule
from typing import Optional, Union
from packaging import version as pkg_version

def initialize_megatron(extra_args_provider=None, args_defaults=None,
                        ignore_unknown_args=False, allow_no_cuda=False):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using ascendspeed for cpu only
    data processing. In general this arg should not be set unless you know 
    what you are doing.
    Returns a function to finalize distributed env initialization 
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not args_defaults:
        args_defaults = {}

    if not allow_no_cuda:
        # Make sure cuda is available.
        assert get_accelerator().is_available(), 'AscendSpeed requires accelerator.'

    # Parse args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(extra_args_provider=extra_args_provider,
                         args_defaults=args_defaults,
                         ignore_unknown_args=ignore_unknown_args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print('> setting random seeds to {} ...'.format(args.seed))
        _set_random_seed(args.seed)

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals    
        set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Initialize memory buffers.
        _initialize_mem_buffs()

        # Auto resume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        # No continuation function
        return None


def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from ascendspeed.data.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


def setup_deepspeed_random_and_activation_checkpointing(args):
    '''Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.
    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.
    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    '''
    num_layers = args.num_layers // args.checkpoint_num_layers
    num_layers = num_layers if args.num_layers % args.checkpoint_num_layers == 0 else num_layers + 1
    if args.split_transformers:
        num_layers *= 2

    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=args.partition_activations,
        contiguous_checkpointing=args.contigious_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=args.checkpoint_in_cpu,
        synchronize=args.synchronize_each_layer,
        profile=args.profile_backward)

    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()

    device_count = get_accelerator().device_count()

    if torch.distributed.is_initialized():
        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...',
                  flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device

            get_accelerator().set_device(device)  # only do so when device_count > 0

        # Call the init process
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port

        if args.deepspeed or args.ds_inference:
            deepspeed.init_distributed()
        else:
            torch.distributed.init_process_group(
                backend=args.distributed_backend,
                world_size=args.world_size, rank=args.rank,
                init_method=init_method)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            parallel_state.initialize_model_parallel(args.tensor_model_parallel_size,
                                                     args.pipeline_model_parallel_size,
                                                     args.virtual_pipeline_model_parallel_size)

    # 设置DeepSpeed ZeRO-R，对activation进行优化
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)


def _init_autoresume():
    """Set auto resume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        # No need to do so for CPU-only case.
        if get_accelerator().device_count() == 0:
            seed = seed_
        else:
            seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if get_accelerator().device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)),
                            global_step=args.iteration)


def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()

    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        mpu.init_checkpointed_activations_memory_buffer()


def _is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or \
                torch.distributed.get_rank() % get_accelerator().device_count() == 0:
            return True
        else:
            return False
    else:
        return True


def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    ver = pkg_version.parse(version_str)
    return ver.major, ver.minor, ver.micro

def initialize(args=None,
               model: torch.nn.Module = None,
               optimizer: Optional[Union[Optimizer, DeepSpeedOptimizerCallable]] = None,
               model_parameters: Optional[torch.nn.Module] = None,
               training_data: Optional[torch.utils.data.Dataset] = None,
               lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
               mpu=None,
               dist_init_required: Optional[bool] = None,
               collate_fn=None,
               config=None,
               config_params=None):
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
            The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.

        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    version_ = version
    version_major_, version_minor_, version_patch_ = _parse_version(version_)
    git_hash_ = git_hash
    git_branch_ = git_branch
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(version_, git_hash_,
                                                                             git_branch_),
             ranks=[0])

    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    assert model is not None, "deepspeed.initialize requires a model"

    global dist
    from deepspeed import comm as dist
    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend, dist_init_required=dist_init_required)

    # Set config using config_params for backwards compat
    if config is None and config_params is not None:
        config = config_params

    # Check for deepscale_config for backwards compat
    if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
        logger.warning("--deepscale_config is deprecated, please use --deepspeed_config")
        if hasattr(args, "deepspeed_config"):
            assert (args.deepspeed_config is
                    None), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
        args.deepspeed_config = args.deepscale_config
        args.deepscale_config = None

    # Check that we have only one config passed
    if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
        assert config is None, "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
        config = args.deepspeed_config
    assert config is not None, "DeepSpeed requires --deepspeed_config to specify configuration file"

    if not isinstance(model, PipelineModule):
        config_class = DeepSpeedConfig(config, mpu)
        if config_class.hybrid_engine.enabled:
            engine = DeepSpeedHybridEngine(args=args,
                                           model=model,
                                           optimizer=optimizer,
                                           model_parameters=model_parameters,
                                           training_data=training_data,
                                           lr_scheduler=lr_scheduler,
                                           mpu=mpu,
                                           dist_init_required=dist_init_required,
                                           collate_fn=collate_fn,
                                           config=config,
                                           config_class=config_class)
        else:
            engine = DeepSpeedEngine(args=args,
                                     model=model,
                                     optimizer=optimizer,
                                     model_parameters=model_parameters,
                                     training_data=training_data,
                                     lr_scheduler=lr_scheduler,
                                     mpu=mpu,
                                     dist_init_required=dist_init_required,
                                     collate_fn=collate_fn,
                                     config=config,
                                     config_class=config_class)
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        mpu = model.mpu()
        config_class = DeepSpeedConfig(config, mpu)
        engine = PipelineEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=mpu,
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config=config,
                                config_class=config_class)
        # Fixed an issue where deepspeed PipelineEngine is incompatible with Sequence Parallel.
        # When Sequence Parallel is used, PipelineEngine cannot be used Otherwise, precision problems occur.
        engine.is_pipe_partitioned = engine.is_pipe_partitioned and not args.sequence_parallel
        engine.is_grad_partitioned = engine.is_grad_partitioned and not args.sequence_parallel

    return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]
    return tuple(return_items)

