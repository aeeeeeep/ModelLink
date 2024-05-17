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

import megatron
from ascendspeed.core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN, fused_layer_norm_affine
from ascendspeed.core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
from ascendspeed.core.fusions.rotary_pos_embedding import RotaryEmbedding_wrapper
from ascendspeed.core.pipeline_parallel.p2p_communication import _batched_p2p_ops
from ascendspeed.core.tensor_parallel.random import _set_cuda_rng_state
from ascendspeed.core.tensor_parallel.cross_entropy import _VocabParallelCrossEntropyForward
from ascendspeed.core.tensor_parallel.layers import VocabParallelEmbeddingForward
from ascendspeed.initialize import _compile_dependencies, set_jit_fusion_options
from ..model import (
    GPTModel, parallel_transformer_init, seq_length_wrapper,
    norm_wrapper, SwitchMLP, state_dict_for_save_checkpoint_wrapper,
    core_attention_wrapper, core_attention_forward, FlashSelfAttention,
    ParallelAttention_wrapper, TransformerLanguageModel__init__,
    ParallelAttentionForward, parallel_transformer_forward, parallel_mlp_init_wrapper,
    rms_norm_init_wrapper, rms_norm_forward
)
from ..core import (vocab_embedding_wrapper, initialize_model_parallel_decorator,
                   destroy_model_parallel_decorator, get_expert_parallel_group,
                   get_expert_parallel_rank, get_expert_model_parallel_rank,
                   get_expert_parallel_world_size, get_expert_model_parallel_world_size,
                   set_expert_model_parallel_rank, set_expert_model_parallel_world_size,
                   RotaryEmbedding_forward, apply_rotary_pos_emb,
                   _build_generic_dataset, _build_document_sample_shuffle_indices)
from ..data import build_pretraining_data_loader
from ..tokenizer import build_tokenizer
from ..arguments import parse_args_decorator, validate_args_decorator
from ..checkpointing import _load_base_checkpoint_wrapper, load_checkpoint_wrapper
from ..initialize import initialize_megatron


def exec_patch():
    patch_megatron_core()
    patch_megatron_noncore()


def patch_megatron_core():
    patch_fusions()
    patch_core_models()
    patch_core_transformers()
    patch_pipeline_parallel()
    patch_tensor_parallel()
    patch_parallel_state()
    patch_datasets()


def patch_megatron_noncore():
    patch_miscellaneous()
    patch_model()
    patch_initialize()
    patch_training()


def patch_fusions():

    # patch_core_fused_layer_norm
    megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction # use torch-npu fused layer norm
    megatron.core.fusions.fused_layer_norm.FastLayerNormFN = FastLayerNormFN # use torch-npu fused layer norm


    # patch_core_fused_softmax
    megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax # use torch-npu npu_scaled_masked_softmax
    megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax # use torch-npu npu_scaled_masked_softmax
    megatron.core.fusions.fused_softmax.ScaledSoftmax = ScaledSoftmax # use torch-npu npu_scaled_masked_softmax
    megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
    megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax


def patch_core_models():
    megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__ = RotaryEmbedding_wrapper(
        megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__) # use torch_npu npu_ratary_mul
    megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward = RotaryEmbedding_forward


def patch_core_transformers():
    megatron.core.transformer.attention.apply_rotary_pos_emb = apply_rotary_pos_emb


def patch_pipeline_parallel():
    from megatron.core import pipeline_parallel
    pipeline_parallel.p2p_communication._batched_p2p_ops = _batched_p2p_ops  # send recv bug


def patch_tensor_parallel():
    megatron.core.tensor_parallel.random._set_cuda_rng_state = _set_cuda_rng_state  # default_generators need replace after set_device
    megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward = _VocabParallelCrossEntropyForward # change masked_target for better performance
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = vocab_embedding_wrapper(
        VocabParallelEmbeddingForward)
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
    megatron.model.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction # use torch-npu fused layer norm
    megatron.model.fused_layer_norm.FastLayerNormFN = FastLayerNormFN # use torch-npu fused layer norm
    megatron.model.fused_layer_norm.fused_layer_norm_affine = fused_layer_norm_affine # use torch-npu fused layer norm


    # patch_fused_softmax
    megatron.model.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax # use torch-npu npu_scaled_masked_softmax
    megatron.model.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax # use torch-npu npu_scaled_masked_softmax
    megatron.model.fused_softmax.ScaledSoftmax = ScaledSoftmax # use torch-npu npu_scaled_masked_softmax
    megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
    megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax


    # patch_rms_norm
    megatron.model.rms_norm.RMSNorm.__init__ = rms_norm_init_wrapper(
        megatron.model.rms_norm.RMSNorm.__init__)
    megatron.model.rms_norm.RMSNorm.forward = rms_norm_forward # use fused_rmsnorm


    # patch_transformer
    megatron.model.transformer.ParallelMLP.__init__ = parallel_mlp_init_wrapper(
        megatron.model.transformer.ParallelMLP.__init__)  # fused swiglu
    megatron.model.transformer.SwitchMLP = SwitchMLP
    megatron.model.transformer.ParallelTransformer.__init__ = parallel_transformer_init
    megatron.model.transformer.ParallelTransformer.forward = parallel_transformer_forward
    megatron.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint \
        = state_dict_for_save_checkpoint_wrapper(
        megatron.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint)
    megatron.model.transformer.ParallelAttention.__init__ = ParallelAttention_wrapper(
        megatron.model.transformer.ParallelAttention.__init__)
    megatron.model.transformer.ParallelAttention.forward = ParallelAttentionForward
    megatron.model.transformer.CoreAttention.__init__ = core_attention_wrapper(
        megatron.model.transformer.CoreAttention.__init__)
    megatron.model.transformer.CoreAttention.forward = core_attention_forward
    megatron.model.transformer.FlashSelfAttention = FlashSelfAttention
    megatron.model.transformer.apply_rotary_pos_emb = apply_rotary_pos_emb

    # patch gptmodel
    megatron.model.GPTModel = GPTModel

    # patch language model
    megatron.model.language_model.TransformerLanguageModel.forward = (seq_length_wrapper(
        megatron.model.language_model.TransformerLanguageModel.forward))
    megatron.model.language_model.TransformerLanguageModel.__init__ = TransformerLanguageModel__init__



def patch_initialize():
    megatron.initialize._compile_dependencies = _compile_dependencies  # remove cuda kernel compile
    megatron.initialize.set_jit_fusion_options = set_jit_fusion_options  # remove cuda jit nvfuser
    megatron.initialize.parse_args = parse_args_decorator(megatron.initialize.parse_args)
    megatron.initialize.initialize_megatron = initialize_megatron


def patch_training():
    from ..training import get_model_wrapper, train
    megatron.training.get_model = get_model_wrapper(megatron.training.get_model)
    megatron.training.build_pretraining_data_loader = build_pretraining_data_loader
    megatron.training.train = train
    megatron.training.load_checkpoint = load_checkpoint_wrapper(megatron.checkpointing.load_checkpoint)


def patch_miscellaneous():
    megatron.arguments.parse_args = parse_args_decorator(megatron.arguments.parse_args)
    megatron.arguments.validate_args = validate_args_decorator(megatron.arguments.validate_args)
    megatron.global_vars.build_tokenizer = build_tokenizer
    megatron.checkpointing._load_base_checkpoint = _load_base_checkpoint_wrapper(
        megatron.checkpointing._load_base_checkpoint)   


def patch_datasets():
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset
    GPTDataset._build_document_sample_shuffle_indices = _build_document_sample_shuffle_indices
    BlendedMegatronDatasetBuilder._build_generic_dataset = _build_generic_dataset
