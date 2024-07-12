import os
import sys
import argparse
from functools import wraps
import torch
from torch_npu.contrib import transfer_to_npu
from .arguments import process_args

_ARGS = None


def get_mindspeed_args():
    global _ARGS
    if _ARGS is None:
        parser = argparse.ArgumentParser(description='MindSpeed Arguments', allow_abbrev=False)
        _ARGS, _ = process_args(parser).parse_known_args()
    return _ARGS


def dummy_jit(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def type_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if isinstance(res, str):
            res = res.replace('npu', 'cuda')
        return res

    return wrapper


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        if name == 'transformer-engine':
            return '0.0'
        res = fn(name, *args, **kwargs)
        return res

    return wrapper


# Patch view method to ensure tensor is contiguous before performing view
def ensure_contiguous_wrapper(fn):
    def wrapper(tensor, *args, **kwargs):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return fn(tensor, *args, **kwargs)

    return wrapper


def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    return op(noop_flag_buffer, tensor_lists, *args)


def multi_tensor_l2norm(overflow_buf, tensor_lists, per_parameter):
    total_norm = 0.0
    norm_type = 2.0
    ret_per_tensor = [] if per_parameter else None
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type
        if per_parameter:
            ret_per_tensor.append(total_norm.clone())
    if not tensor_lists:
        grad_norm = torch.cuda.FloatTensor([0])
        total_norm = grad_norm ** norm_type
    return total_norm ** (1 / norm_type), ret_per_tensor


def multi_tensor_scale(overflow_buf, tensor_lists, scale):
    if len(tensor_lists) != 2:
        raise AssertionError('The size of tensor list must be 2, but got {}'.format(len(tensor_lists)))
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('The size of tensor list must be same, but got {} and {}'.format(len(tensor_lists[0]),
                                                                                              len(tensor_lists[1])))

    with torch.no_grad():
        for i in range(len(tensor_lists[0])):
            tensor_lists[1][i].copy_(tensor_lists[0][i] * scale)


def repeat_interleave(inputs, repeats, dim):
    shape = inputs.shape
    new_shape = shape[:dim + 1] + (repeats,) + shape[dim + 1:]
    out_shape = shape[:dim] + (shape[dim] * repeats,) + shape[dim + 1:]
    return inputs.unsqueeze(dim + 1).expand(new_shape).reshape(out_shape)


def te_adaptation(aspm):
    # Need replace modules before import megatron
    aspm.register_patch('importlib.metadata.version', version_wrapper)
    aspm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
    aspm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)


def apex_adaptation(aspm):
    from .optimizer.adamw import AdamW
    from .core.fusions.fused_layer_norm import fused_layer_norm_affine
    aspm.register_patch('apex.optimizers.FusedAdam', AdamW, create_dummy=True)
    aspm.register_patch('amp_C.multi_tensor_l2norm', multi_tensor_l2norm, create_dummy=True)
    aspm.register_patch('amp_C.multi_tensor_scale', multi_tensor_scale, create_dummy=True)
    aspm.register_patch('fused_layer_norm_cuda', create_dummy=True)
    aspm.register_patch('apex.multi_tensor_apply.multi_tensor_applier', multi_tensor_applier, create_dummy=True)
    aspm.register_patch('apex.normalization.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine, create_dummy=True)


def torch_adaptation(aspm):
    aspm.register_patch('torch.nn.parameter.Parameter.type', type_wrapper)
    aspm.register_patch('torch.Tensor.type', type_wrapper)
    aspm.register_patch('torch.Tensor.view', ensure_contiguous_wrapper)
    aspm.register_patch('torch.Tensor.repeat_interleave', repeat_interleave)


def megatron_core_adaptation(aspm):
    import megatron.core
    megatron.core.jit.jit_fuser = dummy_jit
    from .core.tensor_parallel.random import _set_cuda_rng_state, checkpoint_function_backward
    from .core.tensor_parallel.layers import vocab_parallel_embedding_forward
    from .core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
    from .core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN
    from .core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    from .core.fusions.rotary_pos_embedding import apply_fused_rotary_pos_emb_bshd_wrapper, rotary_embedding_init_wrapper
    from .core.fusions.fused_bias_swiglu import SwiGLUFunction, BiasSwiGLUFunction
    from .core.transformer.attention import attention_init_wrapper
    from .core.tensor_parallel.layers import row_parallel_nocomm_optimizer_wrapper
    from .core.transformer.custom_layers.transformer_engine import PTNorm
    from .core.transformer.dot_product_attention import dot_product_attention_forward_wrapper, dot_product_attention_init_wrapper
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from .core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec_wrapper
    from .core.parallel_state import initialize_model_parallel
    from .core.parallel_state import initialize_model_parallel_wrapper
    from .core.parallel_state import destroy_model_parallel_wrapper
    from .core.memory.auto_pipeline.autopipeline_solver import destroy_model_parallel_profiling_wrapper
    from .core.parallel_state import get_context_parallel_group_for_send_recv_overlap
    from .optimizer.optimizer import (mixed_precision_optimizer_step, \
                                      reuse_fp32_param_init_wrapper, optimizer_config_init_wrapper)
    from .core.pipeline_parallel.schedules import get_forward_backward_func_wrapper
    from .optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
    from .core.models.common.embeddings.rotary_pos_embedding import get_pos_emb_on_this_cp_rank

    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
                        get_pos_emb_on_this_cp_rank)
    aspm.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
                        vocab_parallel_cross_entropy_forward)
    aspm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)
    aspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
                        checkpoint_function_backward)
    aspm.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                        vocab_parallel_embedding_forward)
    aspm.register_patch('megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction',
                        FusedLayerNormAffineFunction)
    aspm.register_patch('megatron.core.fusions.fused_layer_norm.FastLayerNormFN', FastLayerNormFN)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                        ScaledUpperTriangMaskedSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax', ScaledMaskedSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledSoftmax', ScaledSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                        is_kernel_available)
    aspm.register_patch('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
                        forward_fused_softmax)
    aspm.register_patch('megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction', SwiGLUFunction)
    aspm.register_patch('megatron.core.fusions.fused_bias_swiglu.BiasSwiGLUFunction', BiasSwiGLUFunction)
    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.apply_rotary_pos_emb_bshd',
                        apply_fused_rotary_pos_emb_bshd_wrapper)
    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__',
                        rotary_embedding_init_wrapper)
    aspm.register_patch('megatron.core.transformer.attention.Attention.__init__', attention_init_wrapper)
    aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward',
                        row_parallel_nocomm_optimizer_wrapper)

    aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec',
                        get_gpt_layer_local_spec)
    aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec',
                        get_gpt_layer_local_spec_wrapper)
    aspm.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TENorm', PTNorm)
    aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
                        dot_product_attention_init_wrapper)
    aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                        dot_product_attention_forward_wrapper)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                        get_forward_backward_func_wrapper)

    # Bugfix for Megatron-LM core 0.6.0, to be removed for next version.
    aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel', initialize_model_parallel)

    # MoE relative.
    aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                        initialize_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                        destroy_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                        destroy_model_parallel_profiling_wrapper)
    aspm.register_patch('megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                        get_context_parallel_group_for_send_recv_overlap)
    aspm.register_patch('megatron.core.mpu', megatron.core.parallel_state)

    # optim relative.
    aspm.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
                        mixed_precision_optimizer_step)
    aspm.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                        reuse_fp32_param_init_wrapper)
    aspm.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                        optimizer_config_init_wrapper)
    aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                        reuse_fp32_param_distrib_optimizer_init_wrapper)


def megatron_legacy_adaptation(aspm):
    from .core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN, fused_layer_norm_affine
    from .core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    from .core.fusions.rms_norm import rms_norm_init_wrapper, rms_norm_forward_wrapper
    from .model.transformer import parallel_mlp_init_wrapper, flash_self_attention_forward, flash_self_attention_init_wrapper, parallel_mlp_forward_wrapper, parallel_transformer_init_wrapper, \
        parallel_transformer_forward_wrapper
    from .model.transformer import core_attention_init_wrapper, core_attention_forward, parallel_attention_init_wrapper, \
        parallel_attention_forward
    from .core.transformer.transformer import parallel_transformer_layer_forward_wrapper, \
        parallel_transformer_checkpointed_forward_wrapper
    from .model.transformer import switch_mlp_init_wrapper, switch_mlp_forward_wrapper, \
                                    parallel_transformer_layer_init_wrapper
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', parallel_transformer_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.forward', parallel_transformer_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.FusedLayerNormAffineFunction',
                        FusedLayerNormAffineFunction)
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.FastLayerNormFN', FastLayerNormFN)
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                        ScaledUpperTriangMaskedSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledMaskedSoftmax', ScaledMaskedSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledSoftmax', ScaledSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                        is_kernel_available)
    aspm.register_patch('megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
                        forward_fused_softmax)
    aspm.register_patch('megatron.legacy.model.rms_norm.RMSNorm.__init__', rms_norm_init_wrapper)
    aspm.register_patch('megatron.legacy.model.rms_norm.RMSNorm.forward', rms_norm_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelMLP.__init__', parallel_mlp_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelMLP.forward', parallel_mlp_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.FlashSelfAttention.forward', flash_self_attention_forward)
    aspm.register_patch('megatron.legacy.model.transformer.FlashSelfAttention.__init__', flash_self_attention_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelAttention.__init__', parallel_attention_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelAttention.forward',
                        parallel_attention_forward)
    aspm.register_patch('megatron.legacy.model.transformer.CoreAttention.__init__', core_attention_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.CoreAttention.forward', core_attention_forward)

    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformerLayer.forward',
                        parallel_transformer_layer_forward_wrapper)

    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer._checkpointed_forward',
                        parallel_transformer_checkpointed_forward_wrapper)

    aspm.register_patch('megatron.legacy.model.transformer.SwitchMLP.__init__', switch_mlp_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.SwitchMLP.forward', switch_mlp_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformerLayer.__init__',
                        parallel_transformer_layer_init_wrapper)


def megatron_training_adaptation(aspm):
    from .initialize import _compile_dependencies, set_jit_fusion_options_wrapper
    from .utils import get_batch_on_this_cp_rank
    from .training import pretrain
    from .arguments import parse_args_wrapper, validate_args_wrapper, core_transformer_config_from_args_wrapper
    from .tokenizer import build_tokenizer_wrapper
    from .yaml_arguments import core_transformer_config_from_yaml_wrapper
    from .core.training import pretrain_decorator, setup_model_and_optimizer_decorator
    aspm.register_patch('megatron.training.training.pretrain', pretrain_decorator)
    aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_decorator)
    aspm.register_patch('megatron.training.yaml_arguments.core_transformer_config_from_yaml',
                        core_transformer_config_from_yaml_wrapper)
    aspm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
    aspm.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
    aspm.register_patch('megatron.training.arguments.parse_args',
                        parse_args_wrapper)
    aspm.register_patch('megatron.training.arguments.validate_args',
                        validate_args_wrapper)
    aspm.register_patch('megatron.training.yaml_arguments.validate_yaml',
                        validate_args_wrapper)
    aspm.register_patch('megatron.training.arguments.core_transformer_config_from_args',
                        core_transformer_config_from_args_wrapper)
    aspm.register_patch('megatron.training.initialize.set_jit_fusion_options', set_jit_fusion_options_wrapper)
    aspm.register_patch('megatron.training.training.pretrain', pretrain)
    aspm.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer_wrapper)


def ascend_adaptation(aspm):
    from .initialize import coc_registration_wrapper, mc2_wrapper
    from megatron.legacy.model.transformer import ParallelTransformerLayer
    if int(os.getenv('MEMORY_FRAGMENTATION', '0')):
        from .core.memory.memory_fragmentation.pluggable_allocator_adpator import change_allocator
        change_allocator()

        from .core.memory.memory_fragmentation.memory_recorder import memory_recorder_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', memory_recorder_wrapper)

        from .core.memory.memory_fragmentation.malloc_recorder import malloc_recorder_wrapper
        aspm.register_patch('megatron.training.training.train_step', malloc_recorder_wrapper)

        from .core.memory.memory_fragmentation.optimizer_init_precise import optimizer_init_wrapper
        aspm.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step', optimizer_init_wrapper)

        from .core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper
        allowed_recomputing_module_wrapper(ParallelTransformerLayer)
        from .core.memory.adaptive_recomputing.adaptive_recompute import setup_model_and_optimizer_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_wrapper)

    if int(os.getenv('ADAPTIVE_RECOMPUTING', '0')) and not int(os.getenv('MEMORY_FRAGMENTATION', '0')):
        from .core.memory.adaptive_recomputing.pluggable_allocator_adpator import change_allocator
        change_allocator()
        from .core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper
        allowed_recomputing_module_wrapper(ParallelTransformerLayer)
        from .core.memory.adaptive_recomputing.adaptive_recompute import setup_model_and_optimizer_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_wrapper)

    if int(os.getenv('ASCEND_MC2', '0')):
        aspm.register_patch('megatron.training.initialize.initialize_megatron', mc2_wrapper)
    aspm.register_patch('megatron.training.initialize.initialize_megatron', coc_registration_wrapper)

    if int(os.getenv('ADAPTIVE_RECOMPUTING', '0')) or int(os.getenv('MEMORY_FRAGMENTATION', '0')):
        import megatron.training.initialize
        aspm.register_patch('megatron.training.initialize_megatron', megatron.training.initialize.initialize_megatron)


def mcore_moe_adaptation(pm, args):
    if args.moe_permutation_async_comm:
        from .core.transformer.moe.router import aux_loss_load_balancing
        from .core.transformer.moe.token_dispatcher import token_permutation, token_unpermutation
        pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation', token_permutation)
        pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation', token_unpermutation)
        pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing', aux_loss_load_balancing)

    if int(os.getenv('ASCEND_MC2', '0')):
        # MoE MLP not use mc2 linear
        from .core.models.gpt.gpt_layer_specs import get_mlp_module_spec_wrapper
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        import megatron.core.models.gpt.gpt_layer_specs
        megatron.core.models.gpt.gpt_layer_specs._get_mlp_module_spec = get_mlp_module_spec_wrapper(
            megatron.core.models.gpt.gpt_layer_specs._get_mlp_module_spec, ColumnParallelLinear.forward,
            RowParallelLinear.forward)


def exe_adaptation():
    mindspeed_args = get_mindspeed_args()
    from .patch_utils import MindSpeedPatchesManager as aspm
    te_adaptation(aspm)
    apex_adaptation(aspm)
    torch_adaptation(aspm)
    aspm.apply_patches()
    megatron_core_adaptation(aspm)
    megatron_legacy_adaptation(aspm)
    megatron_training_adaptation(aspm)
    ascend_adaptation(aspm)
    mcore_moe_adaptation(aspm, mindspeed_args)
    aspm.apply_patches()

    # accelerate package will check TE on sys.modules，so we need remove this patch
    del sys.modules['transformer_engine']


exe_adaptation()

