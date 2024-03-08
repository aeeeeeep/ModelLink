from functools import wraps
import torch
import megatron
from megatron.model import GPTModel
from megatron.model import transformer
from megatron import get_args
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.model.module import MegatronModule
from megatron.model.enums import AttnMaskType, LayerType
from megatron.model.language_model import parallel_lm_logits
from megatron.model.fused_layer_norm import MixedFusedLayerNorm
from megatron.model.language_model import get_language_model
from megatron.model.transformer import _get_layer_type
from megatron.model.transformer import ParallelTransformer, ParallelTransformerLayer
from megatron.model.utils import get_norm

from .model.moe import SwitchMLP
from .model.module import MegatronModuleForCausalLM


def seq_length_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.seq_length = get_args().seq_length
        return fn(self, *args, **kwargs)

    return wrapper


class BaseModel(GPTModel, MegatronModuleForCausalLM):
    def __init__(self, config, num_tokentypes=0, parallel_output=True, pre_process=True, post_process=True):
        super(BaseModel, self).__init__(config=config, num_tokentypes=num_tokentypes, parallel_output=parallel_output,
                                        pre_process=pre_process, post_process=post_process)


def apply_model_patch():
    megatron.model.GPTModel = GPTModel
    megatron.model.language_model.TransformerLanguageModel.forward = (seq_length_wrapper(
        megatron.model.language_model.TransformerLanguageModel.forward))

    # moe
    megatron.model.transformer.SwitchMLP = SwitchMLP

    # bloom
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__ = norm_wrapper(
        megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__
    )
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = vocab_embedding_wrapper(
        megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward
    )
    megatron.model.transformer.ParallelTransformer.__init__ = parallel_transformer_init


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):
    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0, 1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        if fp16_lm_cross_entropy:
            if output.dtype != torch.half:
                raise ValueError("Wrong output dtype when fp16_lm_cross_entropy.")
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss


class GPTModel(MegatronModule, MegatronModuleForCausalLM):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        super().__init__(config=config,
                         share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None):

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            return post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


def norm_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args = get_args()
        if parallel_state.is_pipeline_first_stage() and args.embed_layernorm:
            norm = MixedFusedLayerNorm(args.hidden_size)
            self.norm = norm
    return wrapper


def vocab_embedding_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        output = fn(self, *args, **kwargs)
        if hasattr(self, 'norm'):
            output = self.norm(output)
        return output
    return wrapper


def assert_judge(expression):
    if not expression:
        raise AssertionError


def get_num_layers(args, model_type, is_decoder=False):
    """Compute the number of transformer layers resident on the current rank."""
    is_encoder_and_decoder_model = (model_type == ModelType.encoder_and_decoder)
    if model_type == ModelType.retro_encoder:
        num_layers = args.retro_encoder_layers
    elif mpu.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            assert_judge(args.pipeline_model_parallel_split_rank is not None)
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
            assert_judge(args.encoder_num_layers % num_ranks_in_encoder == 0)
            assert_judge(args.decoder_num_layers % num_ranks_in_decoder == 0)
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
            assert_judge(args.num_layers == args.encoder_num_layers)
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

        assert not args.squared_relu, "TransformerEngine does not support squared relu activation."

    self.use_fp8 = args.fp8 is not None
    self.fp8_recipe = None
    self.fp8_group = None
    if self.use_fp8:
        assert_judge(args.transformer_impl == 'transformer_engine')
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
    self.num_layers = get_num_layers(args, model_type,
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
        assert_judge(self.recompute_granularity != 'full')
        assert_judge(args.transformer_impl == 'local')

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
        assert_judge(config.num_layers % config.virtual_pipeline_model_parallel_size == 0)
        assert_judge(args.model_type != ModelType.encoder_and_decoder)
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