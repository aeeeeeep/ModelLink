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


"""Pretrain utilities."""
from datetime import datetime
import time
import json

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.compression.compress import redundancy_clean


from ascendspeed.core.pipeline_parallel.fold_adaptor.adaptor_schedules import forward_backward_pipelining_with_foldx_fifo
from ascendspeed.core.pipeline_parallel.fold_adaptor.adaptor_schedules import forward_backward_pipelining_with_foldx_aiao

import ascendspeed.training
from ascendspeed import get_args
from ascendspeed import get_timers
from ascendspeed import get_num_microbatches
from ascendspeed import print_rank_0
from ascendspeed.training import print_datetime, _initialize_optimized_pipeline, train, setup_teacher_model, get_model, \
    setup_model_and_optimizer, evaluate_and_print_results, build_train_valid_test_data_iterators
from ascendspeed.core import parallel_state
from ascendspeed.checkpointing import save_checkpoint
from ascendspeed.initialize import initialize_megatron
from ascendspeed.core.utils import get_model_config
from ascendspeed.utils import throughput_calculator
from ascendspeed.core.pipeline_parallel.schedules import get_forward_backward_func
from ascendspeed.error_utils import check_type
_TRAIN_START_TIME = time.time()


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize ascendspeed.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla, we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g. images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initialize and get arguments, timers, and TensorBoard writer.
    # 1.初始化分布式环境
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time, so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = get_accelerator().FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize ascendspeed (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after ascendspeed is initialized.')

    args = get_args()
    timers = get_timers()

    if args.optimized_pipeline:
        _initialize_optimized_pipeline()

    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, 'r', encoding='utf-8'))            
        if "curriculum_learning" in args.deepspeed_configuration and \
                "enabled" in args.deepspeed_configuration["curriculum_learning"]:
            args.curriculum_learning_legacy = args.deepspeed_configuration[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_configuration["curriculum_learning"])
        if "compression_training" in args.deepspeed_configuration:
            args.compression_training = True

    # Model, optimizer, and learning rate.
    # 2、模型并行：定义模型架构，并切割模型
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        model_provider, model_type, teacher=False, data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    config = get_model_config(model[0])

    # Data stuff.
    # 3、构造train/val/test数据集
    timers('train/valid/test-data-iterators-setup', log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
        if args.foldx_mode is not None:
            train_data_iterator = [[] for _ in all_data_iterators]
            if all_data_iterators[0][0] is None:
                from types import SimpleNamespace
                train_data_iterator[0] = SimpleNamespace()
            else:
                train_data_iterator[0] = all_data_iterators[0][0]
            train_data_iterator[0].dummy_iterators = train_data_iterator[1:]
        valid_data_iterator = [[
            all_data_iterators[i][1][j] for i in range(len(all_data_iterators))]
            for j in range(len(all_data_iterators[0][1]))
        ]
        test_data_iterator = [[
            all_data_iterators[i][2][j] for i in range(len(all_data_iterators))]
            for j in range(len(all_data_iterators[0][2]))
        ]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    if args.data_efficiency_curriculum_learning:
        if args.deepspeed_dataloader is not None:
            # We use args to pass the deepspeed_dataloader because adding
            # output to setup_model_and_optimizer will break the API for other
            # cases. We clear args.deepspeed_dataloader after updating
            # train_data_iterator because args will be saved in checkpoint and
            # attempting to save the whole deepspeed_dataloader will lead to
            # "AttributeError: Can't pickle local object...".
            train_data_iterator = iter(args.deepspeed_dataloader)
            args.deepspeed_dataloader = None
        else:
            train_data_iterator = None
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # args.teacher_model is used as global variable to pass the teacher model
    # for knowledge distillation. Users do not need to set it in the command
    # line to use kd, but users do need to provide teacher model configurations
    # like args.num_layers_teacher as described in setup_teacher_model().
    args.teacher_model = None
    if args.mos or args.kd:  # Set up teacher model.
        args.teacher_model = setup_teacher_model(args, model_provider)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'], barrier=True)
    print_rank_0('training ...')

    # 4、正式训练
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator, config)
    print_datetime('after training is done')

    if args.do_valid:
        prefix = 'the end of training for val data'
        for iterator in valid_data_iterator:
            evaluate_and_print_results(prefix, forward_step_func,
                                       iterator, model,
                                       iteration, False)

    # Clean the model and do evaluation again
    if args.compression_training:
        model = [redundancy_clean(model[0], args.deepspeed_config, parallel_state)]
        if args.do_valid:
            prefix = 'the end of training and after model cleaning for val data'
            for iterator in valid_data_iterator:
                evaluate_and_print_results(prefix, forward_step_func,
                                           iterator, model,
                                           iteration, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        for iterator in test_data_iterator:
            evaluate_and_print_results(prefix, forward_step_func,
                                       iterator, model,
                                       0, True)


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    if args.deepspeed and args.ds_pipeline_enabled:
        skipped_iter = 0
        num_zeros_in_grad = 0
        check_type(model[0], deepspeed.PipelineEngine)
        loss = model[0].train_batch(data_iter=data_iterator)
        grad_norm = model[0].get_global_grad_norm()
        return {'lm loss': loss}, skipped_iter, grad_norm, num_zeros_in_grad

    # Set grad to zero.
    if not args.deepspeed:
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        else:
            optimizer.zero_grad()

    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    forward_backward_func = get_forward_backward_func()

    if args.mos or args.kd:
        # args.teacher_forward is used as global variable to enable kd loss
        # calculation in forward pass. Users do not need to set it in the
        # command line to use kd.
        args.teacher_forward = True
    if forward_backward_func == forward_backward_pipelining_with_foldx_fifo or\
            forward_backward_func == forward_backward_pipelining_with_foldx_aiao:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length)
    else:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    if args.mos or args.kd:
        args.teacher_forward = False
    # reset timers if necessary
    if config.timers is None:
        config.timers = timers
    timers('forward-backward').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce', log_level=1).start(barrier=args.barrier_with_L1_time)
    if not args.deepspeed:
        optimizer.reduce_model_grads(args, timers)
    timers('backward-embedding-all-reduce').stop()

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    if args.deepspeed:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        model[0].step(lr_kwargs={'increment': increment})
        update_successful = model[0].was_step_applied()
    else:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
        if update_successful:
            optimizer.gather_model_params(args, timers)
    timers('optimizer').stop()

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0
        grad_norm = None
        num_zeros_in_grad = None

        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    else:
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            lr_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


ascendspeed.training.pretrain = pretrain
ascendspeed.training.train_step = train_step

