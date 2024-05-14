# coding=utf-8
# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

from functools import wraps

import math
import numpy as np
import torch
import torch_npu
import torch.nn.functional as F

from megatron import get_timers, get_args, get_retro_args, core, get_num_microbatches
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.parallel_state import get_tensor_model_parallel_group

from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.transformer import _get_layer_type
from megatron.model.transformer import (
    ParallelTransformer, ParallelTransformerLayer, NoopTransformerLayer
)
from megatron.model.utils import get_norm

from modellink.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from modellink.error_utils import ensure_valid
from modellink.model.alibi import Alibi, _build_alibi_tensor, _get_inverted_mask
from ..tasks.finetune.lora.utils import is_enable_lora


try:
    from einops import rearrange
except ImportError:
    rearrange = None


def state_dict_for_save_checkpoint(state_dict):
    state_dict_ = dict()
    for key in state_dict:
        if 'lora' in key:
            state_dict_[key] = state_dict[key]
    return state_dict_


def state_dict_for_save_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(self, prefix='', keep_vars=False):
        if is_enable_lora():
            return state_dict_for_save_checkpoint(self.state_dict(prefix=prefix, keep_vars=keep_vars))
        return fn(self, prefix='', keep_vars=False)

    return wrapper


def _get_num_layers(args, model_type, is_decoder=False):
    """Compute the number of transformer layers resident on the current rank."""
    is_encoder_and_decoder_model = (model_type == ModelType.encoder_and_decoder)
    if model_type == ModelType.retro_encoder:
        num_layers = args.retro_encoder_layers
    elif mpu.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            ensure_valid(args.pipeline_model_parallel_split_rank is not None)
            # When a standalone embedding stage is used, a rank is taken from
            # the encoder's ranks, to be used for the encoder's embedding
            # layer. This way, the rank referenced by the 'split rank' remains
            # the same whether or not a standalone embedding stage is used.
            num_ranks_in_encoder = (
                args.pipeline_model_parallel_split_rank - 1
                if args.standalone_embedding_stage else
                args.pipeline_model_parallel_split_rank
            )
            num_ranks_in_decoder = args.transformer_pipeline_model_parallel_size - num_ranks_in_encoder
            ensure_valid(args.encoder_num_layers % num_ranks_in_encoder == 0)
            ensure_valid(args.decoder_num_layers % num_ranks_in_decoder == 0)
            if mpu.is_pipeline_stage_before_split():
                num_layers = (
                    0
                    if args.standalone_embedding_stage
                    and mpu.get_pipeline_model_parallel_rank() == 0 else
                    args.encoder_num_layers // num_ranks_in_encoder
                )
            else:
                num_layers = args.decoder_num_layers // num_ranks_in_decoder
        else:
            ensure_valid(args.num_layers == args.encoder_num_layers)
            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if args.standalone_embedding_stage
                and mpu.get_pipeline_model_parallel_rank() == 0 else
                args.num_layers // args.transformer_pipeline_model_parallel_size
            )
    else:
        if not is_decoder:
            num_layers = args.encoder_num_layers
        else:
            num_layers = args.decoder_num_layers
    return num_layers


def parallel_transformer_init(self, config,
                 model_type, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0):
    super(ParallelTransformer, self).__init__()
    args = get_args()

    self.layer_type = layer_type
    self.model_type = model_type
    self.bf16 = config.bf16
    self.fp32_residual_connection = config.fp32_residual_connection
    self.post_norm = post_norm
    self.pre_process = pre_process
    self.post_process = post_process
    self.input_tensor = None
    self.drop_path_rate = drop_path_rate
    self.transformer_impl = args.transformer_impl
    self.retro_add_retriever = args.retro_add_retriever

    # Store activation checkpoiting flag.
    self.recompute_granularity = config.recompute_granularity
    self.recompute_method = config.recompute_method
    self.recompute_num_layers = config.recompute_num_layers
    self.distribute_saved_activations = \
        config.distribute_saved_activations and not config.sequence_parallel

    self.sequence_parallel = config.sequence_parallel

    # Transformer Engine Init.
    self.transformer_engine_v_0_10 = False
    self.transformer_engine_v_0_11 = False
    self.transformer_engine_v_0_8 = False
    if self.transformer_impl == 'transformer_engine':
        global transformer_engine
        import transformer_engine
        from importlib.metadata import version
        from pkg_resources import packaging

        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.8.0"):
            self.transformer_engine_v_0_8 = True
        if te_version >= packaging.version.Version("0.10.0"):
            self.transformer_engine_v_0_10 = True
        if te_version >= packaging.version.Version("0.11.0"):
            self.transformer_engine_v_0_11 = True

        del version, packaging

        ensure_valid(not args.squared_relu)

    self.use_fp8 = args.fp8 is not None
    self.fp8_recipe = None
    self.fp8_group = None
    if self.use_fp8:
        ensure_valid(args.transformer_impl == 'transformer_engine')
        self.fp8_group = mpu.get_amax_reduction_group()
        if args.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif args.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("The DelayedScaling recipe only supports E4M3 and HYBRID formats.")
        self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
            margin=args.fp8_margin,
            interval=args.fp8_interval,
            fp8_format=fp8_format,
            amax_history_len=args.fp8_amax_history_len,
            amax_compute_algo=args.fp8_amax_compute_algo,
            override_linear_precision=(False, False, not args.fp8_wgrad),
        )

    self.num_microbatches_in_previous_step = -1
    self.microbatch_count = 0
    self.checkpoint_core_attention = config.recompute_granularity == 'selective'

    # Number of layers.
    self.num_layers = _get_num_layers(args, model_type,
                                      layer_type == LayerType.decoder)

    self.drop_path_rates = []
    for rate in torch.linspace(0, self.drop_path_rate, config.num_layers):
        self.drop_path_rates.append(rate.item())

    self.retro_layer_numbers = None
    if model_type == ModelType.retro_decoder:
        retro_layer_start = 6 if config.num_layers <= 15 else 9
        self.retro_layer_numbers = \
            np.arange(retro_layer_start, args.num_layers + 1, 3).tolist()
    if model_type == ModelType.retro_encoder:
        self.retro_layer_numbers = [1]

    # Transformer layers.
    if args.retro_add_retriever:
        ensure_valid(self.recompute_granularity != 'full')
        ensure_valid(args.transformer_impl == 'local')

    def build_layer(layer_number):
        if args.transformer_impl == 'local':
            current_layer_type = _get_layer_type(
                model_type, layer_type, self.retro_layer_numbers,
                layer_number)
            return ParallelTransformerLayer(
                config,
                layer_number,
                layer_type=current_layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])
        else:
            # This argument is only available from TE v0.10 onwards.
            extra_transformer_engine_kwargs = {}
            if self.transformer_engine_v_0_8:
                extra_transformer_engine_kwargs["bias"] = args.add_bias_linear
            if self.transformer_engine_v_0_10:
                extra_transformer_engine_kwargs["activation"] = "swiglu" if args.swiglu else "gelu"
            if self.transformer_engine_v_0_11:
                extra_transformer_engine_kwargs["normalization"] = args.normalization
            return transformer_engine.pytorch.TransformerLayer(
                config.hidden_size,
                config.ffn_hidden_size,
                config.num_attention_heads,
                layernorm_epsilon=config.layernorm_epsilon,
                hidden_dropout=config.hidden_dropout,
                attention_dropout=config.attention_dropout,
                init_method=config.init_method,
                output_layer_init_method=config.output_layer_init_method,
                layer_number=layer_number,
                kv_channels=config.kv_channels,
                self_attn_mask_type=self_attn_mask_type.name,
                tp_group=mpu.get_tensor_model_parallel_group(),
                get_rng_state_tracker=tensor_parallel.get_cuda_rng_tracker,
                fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
                apply_query_key_layer_scaling=config.apply_query_key_layer_scaling,
                attention_softmax_in_fp32=config.attention_softmax_in_fp32,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                sequence_parallel=config.sequence_parallel,
                params_dtype=config.params_dtype,
                apply_residual_connection_post_layernorm=config.apply_residual_connection_post_layernorm,
                output_layernorm=False,
                layer_type="encoder",
                drop_path_rate=self.drop_path_rates[layer_number - 1],
                set_parallel_mode=True,
                fuse_qkv_params=True,
                **extra_transformer_engine_kwargs)

    if config.virtual_pipeline_model_parallel_size is not None:
        ensure_valid(config.num_layers % config.virtual_pipeline_model_parallel_size == 0)
        ensure_valid(args.model_type != ModelType.encoder_and_decoder)
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]
        offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
            config.num_layers // config.virtual_pipeline_model_parallel_size) + \
            (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
    else:
        # Each stage gets a contiguous set of layers.
        if args.model_type == ModelType.encoder_and_decoder and \
                mpu.get_pipeline_model_parallel_world_size() > 1:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
            if layer_type == LayerType.encoder:
                offset = pipeline_rank * self.num_layers
            else:
                num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
        else:
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

    if self.num_layers == 0:
        # When a standalone embedding stage is used (e.g.,
        # args.standalone_embedding_stage == True), virtual pipeline ranks
        # on pipeline rank 0 will have zero transformer layers assigned to
        # them. This results in the model's input and output tensors to be
        # the same, which will cause failure for certain output tensor
        # optimizations (e.g., pipeline output deallocation). To remedy
        # this, we assign a 'no-op' layer on these ranks, which will
        # disconnect the input tensor from the output tensor.
        self.num_layers = 1
        self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
    else:
        # Build the layers
        self.layers = []
        if args.num_layer_list:
            num_layer_list = list(map(int, args.num_layer_list.split(',')))
            start_layer_num = 1
            for idx, value in enumerate(num_layer_list):
                if parallel_state.get_pipeline_model_parallel_rank() == idx:
                    self.num_layers = value
                    for layer_num in range(start_layer_num, start_layer_num + value):
                        self.layers.append(build_layer(layer_num))
                start_layer_num += value
            self.layers = torch.nn.ModuleList(self.layers)
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        # Update dropout rate for Retro encoder.
        if model_type == ModelType.retro_encoder:
            for layer in self.layers:
                if layer.self_attention.use_flash_attn:
                    layer.self_attention.core_attention_flash.dropout_p = \
                        torch.nn.Dropout(args.retro_encoder_attention_dropout)
                else:
                    layer.self_attention.core_attention.attention_dropout.p =\
                        args.retro_encoder_attention_dropout
                layer.hidden_dropout = args.retro_encoder_hidden_dropout

    if self.post_process and self.post_norm:
        # Final layer norm before output.
        self.final_norm = get_norm(config)


def ParallelAttention_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        config = args[0]
        query_projection_size = config.kv_channels * config.num_attention_heads
        _args = get_args()
        if _args.group_query_attention:
            kv_projection_size = _args.kv_channels * _args.num_query_groups
        else:
            kv_projection_size = _args.kv_channels * _args.num_attention_heads
        # qkv bias
        bias = _args.add_qkv_bias or _args.add_bias_linear
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=False)
        # dense bias
        bias = _args.add_dense_bias or _args.add_bias_linear
        skip_bias_add = _args.skip_bias_add
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add)

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=config.attention_dropout,
                layer_number=self.layer_number
            )
    return wrapper


class FlashSelfAttention(torch.nn.Module):
    """ Ascend Flash Attention, support Alibi
    Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 layer_number=None, device=None, dtype=None):
        super().__init__()
        args = get_args()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.pre_tockens = args.pre_tockens
        self.next_tockens = args.next_tockens

        self.square_alibi_mask = args.square_alibi_mask
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.layer_number = layer_number
        self.fill_neg_inf = args.fill_neg_inf

        self.beta = 1.0
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number

        self.alibi = None
        if args.position_embedding_type == 'alibi':
            self.get_alibi()

    def forward(self, q, k, v, pse=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        args = get_args()

        batch_size, seq_length, head_num, head_dim = q.shape[1], q.shape[0], q.shape[2], q.shape[3]

        use_sliding_windows = (
            getattr(args, "sliding_window", None) is not None
            and seq_length > args.sliding_window
            and not self.alibi
        )

        if use_sliding_windows:
            pse = None
            sparse_mode = 4
            self.pre_tockens = args.sliding_window
        else:
            sparse_mode = 0

        if not hasattr(self, 'attention_mask') or self.attention_mask.shape[0] != seq_length:
            if use_sliding_windows:
                self.attention_mask = torch.triu(torch.ones(2048, 2048), 1).bool().npu()
            else:
                self.attention_mask = torch.triu(torch.ones(seq_length, seq_length), 1).bool().npu()


        q, k, v = [rearrange(x, 's b h d -> s b (h d)') for x in [q, k, v]]

        try:
            if not hasattr(self, 'num_factor'):
                self.norm_factor = math.sqrt(head_dim)

            if self.apply_query_key_layer_scaling:
                coeff = self.layer_number
                self.norm_factor *= coeff

            scale = 1.0 / self.norm_factor if self.softmax_scale is None else self.softmax_scale
        except Exception as e:
            raise ValueError('Invalid head_dim: {}'.format(head_dim)) from e

        size_record = q.shape
        if self.alibi is not None and (self.alibi.output_size != size_record) and pse is None:
            if args.shape_order != 'SBH':
                raise ValueError(
                    'FlashAttention with Alibi requires for SBH shape_order, but is {}.'.format(args.shape_order))

            self.alibi.output_size = size_record
            self.get_alibi()
            if self.fill_neg_inf:
                _alibi = self.alibi.alibi[:, :seq_length, :seq_length]
                self.attention_mask = \
                    self.attention_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)[:batch_size, :, :, :]
                self.alibi.matmul_result = _get_inverted_mask(self.attention_mask, _alibi).view(-1, seq_length, seq_length)
            else:
                self.alibi.matmul_result = self.alibi.alibi[:, :, :seq_length].repeat(batch_size, 1, 1)

        if self.alibi and pse is None:
            pse = self.alibi.matmul_result.reshape(batch_size, head_num, seq_length,
                                                   seq_length) * self.beta * self.norm_factor
            self.pre_tockens = seq_length

        output = torch_npu.npu_fusion_attention( \
            q, k, v, head_num, args.shape_order, \
            pse=pse, \
            sparse_mode=sparse_mode, \
            padding_mask=None, \
            atten_mask=self.attention_mask, \
            scale=scale, \
            pre_tockens=self.pre_tockens, \
            next_tockens=self.next_tockens, \
            keep_prob=1 - self.dropout_p, \
            inner_precise=0
        )[0]

        return output

    def get_alibi(self):
        args = get_args()
        self.alibi = Alibi()
        alibi = _build_alibi_tensor(args.seq_length,
                                    args.num_attention_heads,
                                    args.square_alibi_mask,
                                    args.fill_neg_inf,
                                    ).to(torch.cuda.current_device())
        if args.params_dtype == torch.float16:
            alibi = alibi.to(torch.float16)
        elif args.params_dtype == torch.bfloat16:
            alibi = alibi.to(torch.bfloat16)
        self.alibi.alibi = alibi



def core_attention_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)

        args = get_args()
        self.square_alibi_mask = args.square_alibi_mask
        self.fill_neg_inf = args.fill_neg_inf
        self.beta = 1.0
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number
        if args.position_embedding_type == 'alibi':
            self.alibi = Alibi()
            alibi = _build_alibi_tensor(args.seq_length,
                                        args.num_attention_heads,
                                        args.square_alibi_mask,
                                        args.fill_neg_inf
                                        ).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)
            self.alibi.alibi = alibi
        else:
            self.alibi = None

    return wrapper


def core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1),
                   query_layer.size(2),
                   query_layer.size(0),
                   key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.reshape(output_size[2],
                                      output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3],
                               output_size[0] * output_size[1], -1)

    if self.alibi is None:
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),
            key_layer.transpose(0, 1).transpose(1, 2),
            beta=0.0, alpha=(1.0 / self.norm_factor))
    else:
        if self.alibi.matmul_result is None or self.alibi.output_size != output_size:
            args = get_args()

            self.alibi.output_size = output_size
            alibi = _build_alibi_tensor(args.seq_length,
                                        args.num_attention_heads,
                                        args.square_alibi_mask,
                                        args.fill_neg_inf
                                        ).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)
            self.alibi.alibi = alibi

            if self.fill_neg_inf:
                _alibi = self.alibi.alibi[:, :output_size[3], :output_size[3]]
                attention_mask = attention_mask.repeat(output_size[0], 1, 1, 1)[:output_size[0], :, :, :]
                self.alibi.matmul_result = _get_inverted_mask(attention_mask, _alibi).view(-1, output_size[2],
                                                                                           output_size[2]).contiguous()
            else:
                self.alibi.matmul_result = self.alibi.alibi[:, :, :output_size[3]].repeat(output_size[0], 1, 1)

        q_trans = query_layer.transpose(0, 1).contiguous()
        k_trans = key_layer.transpose(0, 1).transpose(1, 2).contiguous()
        matmul_result = self.beta * self.alibi.matmul_result + torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)

        # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    if self.square_alibi_mask:
        attention_scores = torch.max(
            attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min)
        )
        attention_probs = torch.nn.functional.softmax(attention_scores, -1)
    else:
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    if not self.sequence_parallel:
        with tensor_parallel.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)
    else:
        attention_probs = self.attention_dropout(attention_probs)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1),
                   value_layer.size(2),
                   query_layer.size(0),
                   value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0),
                                   output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                           output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + \
                              (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


def ParallelAttentionForward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    is_first_step = False
    if inference_params:
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_len = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)
            inference_value_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)

            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory, inference_value_memory)
            is_first_step = True
        else:
            inference_key_memory, inference_value_memory = \
                inference_params.key_value_memory_dict[self.layer_number]

    # =====================
    # Query, Key, and Value
    # =====================
    if self.attention_type == AttnType.self_attn:

        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
        key_layer,
        value_layer) = torch.split(
            mixed_x_layer,
            [
                (
                    self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head
            ],
            dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
    else:
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(encoder_output)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer,
        value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query_layer, _ = self.query(hidden_states)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_tensor_shape)

    # ==================================
    # Adjust key and value for inference
    # ==================================

    # duplicate the pos_emb for self attention
    if rotary_pos_emb is not None:
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb
        else:
            rotary_pos_emb = ((rotary_pos_emb,) * 2)

    if inference_params:
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key_layer.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key_layer.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end,
                             batch_start:batch_end, ...] = key_layer
        inference_value_memory[sequence_start:sequence_end,
                               batch_start:batch_end, ...] = value_layer
        key_layer = inference_key_memory[
            :sequence_end, batch_start:batch_end, ...]
        value_layer = inference_value_memory[
            :sequence_end, batch_start:batch_end, ...]


        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

    # ==================================
    # core attention computation
    # ==================================

    # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
    if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
        key_layer = key_layer.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim = 2
        )
        value_layer = value_layer.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim = 2
        )

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

    if not self.use_flash_attn:
        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)
    else:
        q, k, v = query_layer, key_layer, value_layer
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(q, k, v)
        else:
            context_layer = self.core_attention_flash(q, k, v)

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.dense(context_layer)

    return output, bias