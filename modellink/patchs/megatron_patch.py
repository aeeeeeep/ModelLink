# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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
import argparse

import megatron
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
import megatron.core.models.gpt.gpt_layer_specs
from mindspeed.core.fusions.fused_layer_norm import (FusedLayerNormAffineFunction, FastLayerNormFN,
                                                     fused_layer_norm_affine)
from mindspeed.core.fusions.fused_softmax import (is_kernel_available, ScaledUpperTriangMaskedSoftmax,
                                                  ScaledMaskedSoftmax, ScaledSoftmax, forward_fused_softmax)

from mindspeed.model.transformer import parallel_mlp_init_wrapper
from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state
from mindspeed.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
from mindspeed.core.tensor_parallel.layers import vocab_parallel_embedding_forward
from mindspeed.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed.core.transformer.moe.router import aux_loss_load_balancing
from mindspeed.core.transformer.moe.token_dispatcher import token_permutation, token_unpermutation
from mindspeed.initialize import _compile_dependencies
from mindspeed.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_wrapper

from ..model import (
    GPTModel, parallel_transformer_init, seq_length_wrapper,
    norm_wrapper, SwitchMLP, state_dict_for_save_checkpoint_wrapper,
    core_attention_wrapper, core_attention_forward, FlashSelfAttention,
    ParallelAttention_wrapper, transformer_language_model_init,
    ParallelAttentionForward, parallel_transformer_forward, parallel_mlp_init_wrapper,
    rms_norm_init_wrapper, rms_norm_forward
)
from ..core import (vocab_embedding_wrapper, initialize_model_parallel_decorator,
                   destroy_model_parallel_decorator, get_expert_parallel_group,
                   get_expert_parallel_rank, get_expert_model_parallel_rank,
                   get_expert_parallel_world_size, get_expert_model_parallel_world_size,
                   set_expert_model_parallel_rank, set_expert_model_parallel_world_size,
                   build_generic_dataset, _build_document_sample_shuffle_indices,
                   topk_router_forward, topk_router_routing, z_loss_func,
                   TransformerLayerSubmodules, transformer_layer_init_wrapper,
                   transformer_layer_forward, gpt_model_forward,
                   get_gpt_layer_local_spec_wrapper,
                   start_grad_sync_wrapper, distributed_data_parallel_init_wrapper,
                   get_megatron_optimizer_wrapper, clip_grad_norm_fp32_wrapper, distributed_optimizer_init_wrapper)
from ..core.pipeline_parallel.p2p_communication import _batched_p2p_ops
from ..data import build_pretraining_data_loader
from ..tokenizer import build_tokenizer
from ..arguments import parse_args_decorator, validate_args_decorator
from ..checkpointing import _load_base_checkpoint_wrapper, load_checkpoint_wrapper
from ..initialize import initialize_megatron
from ..log_handler import emit
from ..arguments import process_args

_ARGS = None


def get_modellink_args():
    """
    获取modellink的参数
    """
    global _ARGS
    if _ARGS is None:
        parser = argparse.ArgumentParser(description='ModelLink Arguments', allow_abbrev=False)
        _ARGS, _ = process_args(parser).parse_known_args()
    return _ARGS


def exec_adaptation():
    patch_megatron_core()
    patch_megatron_noncore()


def patch_megatron_core():
    # 获取参数，提供core models入参
    modellink_args = get_modellink_args()
    patch_fusions()
    patch_core_models(modellink_args)
    patch_core_transformers()
    patch_pipeline_parallel()
    patch_tensor_parallel()
    patch_parallel_state()
    patch_datasets()
    patch_utils()


def patch_megatron_noncore():
    patch_miscellaneous()
    patch_model()
    patch_initialize()
    patch_training()
    patch_log_handler()
    patch_high_availability_feature()


def patch_fusions():
    # patch_core_fused_layer_norm
    megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction  # use torch-npu fused layer norm
    megatron.core.fusions.fused_layer_norm.FastLayerNormFN = FastLayerNormFN  # use torch-npu fused layer norm
    # patch_core_fused_softmax
    megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax  # use torch-npu npu_scaled_masked_softmax
    megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax  # use torch-npu npu_scaled_masked_softmax
    megatron.core.fusions.fused_softmax.ScaledSoftmax = ScaledSoftmax  # use torch-npu npu_scaled_masked_softmax
    megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
    megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax


def patch_core_models(args):
    from mindspeed.core.fusions.rotary_pos_embedding import rotary_embedding_init_wrapper
    from ..core import RotaryEmbedding_forward
    megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward = RotaryEmbedding_forward
    megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__ = \
        rotary_embedding_init_wrapper(
            megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__)
    megatron.core.models.gpt.gpt_model.GPTModel.forward = gpt_model_forward
    megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec = get_gpt_layer_local_spec_wrapper(
        megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec)
    if args.moe_permutation_async_comm:
        megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation = token_permutation
        megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation = token_unpermutation
        megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing = aux_loss_load_balancing
    if args.use_mc2:
        megatron.core.models.gpt.gpt_layer_specs._get_mlp_module_spec = get_mlp_module_spec_wrapper(
            megatron.core.models.gpt.gpt_layer_specs._get_mlp_module_spec, ColumnParallelLinear.forward,
            RowParallelLinear.forward)


def patch_core_transformers():
    from ..core import apply_rotary_pos_emb_bshd_wrapper
    megatron.core.models.common.embeddings.rotary_pos_embedding.apply_rotary_pos_emb_bshd = \
        apply_rotary_pos_emb_bshd_wrapper(
            megatron.core.models.common.embeddings.rotary_pos_embedding.apply_rotary_pos_emb_bshd)
    megatron.core.transformer.transformer_block.TENorm = PTNorm
    megatron.core.transformer.moe.router.TopKRouter.routing = topk_router_routing
    megatron.core.transformer.moe.router.TopKRouter.forward = topk_router_forward
    megatron.core.transformer.transformer_layer.TransformerLayerSubmodules = TransformerLayerSubmodules
    megatron.core.transformer.transformer_layer.TransformerLayer.__init__ = transformer_layer_init_wrapper(
        megatron.core.transformer.transformer_layer.TransformerLayer.__init__)
    megatron.core.transformer.transformer_layer.TransformerLayer.forward = transformer_layer_forward
    megatron.core.transformer.moe.router.z_loss_func = z_loss_func


def patch_pipeline_parallel():
    from megatron.core import pipeline_parallel
    pipeline_parallel.p2p_communication._batched_p2p_ops = _batched_p2p_ops  # send recv bug


def patch_tensor_parallel():
    megatron.core.tensor_parallel.random._set_cuda_rng_state = _set_cuda_rng_state  # default_generators need replace after set_device
    megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward = vocab_parallel_cross_entropy_forward  # change masked_target for better performance
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = vocab_embedding_wrapper(
        vocab_parallel_embedding_forward)
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__ = norm_wrapper(
        megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__)


def patch_parallel_state():
    setattr(megatron.core.parallel_state, "get_expert_parallel_group", get_expert_parallel_group)
    setattr(megatron.core.parallel_state, "get_expert_parallel_rank", get_expert_parallel_rank)
    setattr(megatron.core.parallel_state, "get_expert_model_parallel_rank", get_expert_model_parallel_rank)
    setattr(megatron.core.parallel_state, "get_expert_parallel_world_size", get_expert_parallel_world_size)
    setattr(megatron.core.parallel_state, "get_expert_model_parallel_world_size", get_expert_model_parallel_world_size)
    setattr(megatron.core.parallel_state, "set_expert_model_parallel_rank", set_expert_model_parallel_rank)
    setattr(megatron.core.parallel_state, "set_expert_model_parallel_world_size", set_expert_model_parallel_world_size)
    megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorator(
        megatron.core.parallel_state.initialize_model_parallel)
    megatron.core.parallel_state.destroy_model_parallel = destroy_model_parallel_decorator(
        megatron.core.parallel_state.destroy_model_parallel)


def patch_model():
    # patch_fused_layer_norm
    megatron.legacy.model.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction  # use torch-npu fused layer norm
    megatron.legacy.model.fused_layer_norm.FastLayerNormFN = FastLayerNormFN  # use torch-npu fused layer norm
    megatron.legacy.model.fused_layer_norm.fused_layer_norm_affine = fused_layer_norm_affine  # use torch-npu fused layer norm

    # patch_fused_softmax
    megatron.legacy.model.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax  # use torch-npu npu_scaled_masked_softmax
    megatron.legacy.model.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax  # use torch-npu npu_scaled_masked_softmax
    megatron.legacy.model.fused_softmax.ScaledSoftmax = ScaledSoftmax  # use torch-npu npu_scaled_masked_softmax
    megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
    megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax

    # patch_rms_norm
    megatron.legacy.model.rms_norm.RMSNorm.__init__ = rms_norm_init_wrapper(
        megatron.legacy.model.rms_norm.RMSNorm.__init__)  # use fused_rmsnorm
    megatron.legacy.model.rms_norm.RMSNorm.forward = rms_norm_forward  # use fused_rmsnorm

    # patch_transformer
    megatron.legacy.model.transformer.ParallelMLP.__init__ = parallel_mlp_init_wrapper(
        megatron.legacy.model.transformer.ParallelMLP.__init__)  # fused swiglu
    megatron.legacy.model.transformer.SwitchMLP = SwitchMLP
    megatron.legacy.model.transformer.ParallelTransformer.__init__ = parallel_transformer_init
    megatron.legacy.model.transformer.ParallelTransformer.forward = parallel_transformer_forward
    megatron.legacy.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint \
        = state_dict_for_save_checkpoint_wrapper(
        megatron.legacy.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint)
    megatron.legacy.model.transformer.ParallelAttention.__init__ = ParallelAttention_wrapper(
        megatron.legacy.model.transformer.ParallelAttention.__init__)
    megatron.legacy.model.transformer.ParallelAttention.forward = ParallelAttentionForward
    megatron.legacy.model.transformer.CoreAttention.__init__ = core_attention_wrapper(
        megatron.legacy.model.transformer.CoreAttention.__init__)
    megatron.legacy.model.transformer.CoreAttention.forward = core_attention_forward
    megatron.legacy.model.transformer.FlashSelfAttention = FlashSelfAttention

    # patch gptmodel
    megatron.legacy.model.GPTModel = GPTModel

    # patch language model
    megatron.legacy.model.language_model.TransformerLanguageModel.forward = (seq_length_wrapper(
        megatron.legacy.model.language_model.TransformerLanguageModel.forward))
    megatron.legacy.model.language_model.TransformerLanguageModel.__init__ = transformer_language_model_init


def patch_initialize():
    megatron.training.initialize._compile_dependencies = _compile_dependencies  # remove cuda kernel compile
    megatron.training.initialize.parse_args = parse_args_decorator(megatron.training.initialize.parse_args)
    megatron.training.initialize.initialize_megatron = initialize_megatron


def patch_training():
    from ..training import get_model_wrapper, train
    megatron.training.training.get_model = get_model_wrapper(megatron.training.training.get_model)
    megatron.training.training.build_pretraining_data_loader = build_pretraining_data_loader
    megatron.training.training.train = train
    megatron.training.training.load_checkpoint = load_checkpoint_wrapper(
        megatron.training.checkpointing.load_checkpoint)


def patch_miscellaneous():
    megatron.training.arguments.parse_args = parse_args_decorator(megatron.training.arguments.parse_args)
    megatron.training.arguments.validate_args = validate_args_decorator(megatron.training.arguments.validate_args)
    megatron.training.global_vars.build_tokenizer = build_tokenizer
    megatron.training.checkpointing._load_base_checkpoint = _load_base_checkpoint_wrapper(
        megatron.training.checkpointing._load_base_checkpoint)


def patch_datasets():
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset
    GPTDataset._build_document_sample_shuffle_indices = _build_document_sample_shuffle_indices
    BlendedMegatronDatasetBuilder.build_generic_dataset = build_generic_dataset


def patch_log_handler():
    from megatron.training.log_handler import CustomHandler
    CustomHandler.emit = emit


def patch_utils():
    from ..utils import unwrap_model_wrapper
    megatron.training.checkpointing.unwrap_model = unwrap_model_wrapper(megatron.training.checkpointing.unwrap_model)
    megatron.training.training.unwrap_model = unwrap_model_wrapper(megatron.training.training.unwrap_model)


def patch_high_availability_feature():
    from ..training import setup_model_and_optimizer_wrapper
    megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__ = distributed_data_parallel_init_wrapper(
        megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__)
    megatron.core.distributed.param_and_grad_buffer.Bucket.start_grad_sync = start_grad_sync_wrapper(
        megatron.core.distributed.param_and_grad_buffer.Bucket.start_grad_sync)
    megatron.training.training.get_megatron_optimizer = get_megatron_optimizer_wrapper(
        megatron.training.training.get_megatron_optimizer)
    megatron.core.optimizer.optimizer.clip_grad_norm_fp32 = clip_grad_norm_fp32_wrapper(
        megatron.core.optimizer.optimizer.clip_grad_norm_fp32)
    megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__ = distributed_optimizer_init_wrapper(
        megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__)
    megatron.training.training.setup_model_and_optimizer = setup_model_and_optimizer_wrapper(
        megatron.training.training.setup_model_and_optimizer)
