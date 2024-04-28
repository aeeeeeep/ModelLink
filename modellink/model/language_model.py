# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION and Nvidia Megatron-LM Team.
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

from megatron.training import get_args
from megatron.legacy.model.enums import AttnMaskType
from megatron.legacy.model.language_model import TransformerLanguageModel, Embedding
from megatron.legacy.model.transformer import ParallelTransformer
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core import tensor_parallel


def seq_length_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.seq_length = get_args().seq_length
        return fn(self, *args, **kwargs)

    return wrapper


def TransformerLanguageModel__init__(self,
                 config,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True):
    args = get_args()

    if args.untie_embeddings_and_output_weights:
        assert not add_decoder
    super(TransformerLanguageModel, self).__init__(share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

    self.pre_process = pre_process
    self.post_process = post_process
    self.hidden_size = config.hidden_size
    self.num_tokentypes = num_tokentypes
    self.init_method = config.init_method
    self.add_encoder = add_encoder
    self.encoder_attn_mask_type = encoder_attn_mask_type
    self.add_decoder = add_decoder
    self.decoder_attn_mask_type = decoder_attn_mask_type
    self.add_pooler = add_pooler
    self.encoder_hidden_state = None
    self.add_retriever = args.retro_add_retriever
    self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

    # Embeddings.
    if self.pre_process:
        self.embedding = Embedding(self.hidden_size,
                                    args.padded_vocab_size,
                                    args.max_position_embeddings,
                                    args.hidden_dropout,
                                    config,
                                    self.num_tokentypes)
        self._embedding_key = 'embedding'

    # Rotary positional embeddings
    self.use_rotary_position_embeddings = \
        args.position_embedding_type == 'rope'
    if self.use_rotary_position_embeddings:
        self.seq_length = args.seq_length
        rotary_dim = args.hidden_size // args.num_attention_heads \
            if args.kv_channels is None else args.kv_channels

        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al
        # https://github.com/kingoflolz/mesh-transformer-jax/
        if args.use_partial_rope:
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_dim // 2,
                args.rotary_percent,
                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
            )
        else:
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_dim,
                args.rotary_percent,
                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
            )

    # Encoder (usually set to True, False if part of an encoder-decoder
    # architecture and in encoder-only stage).
    if self.add_encoder:
        self.encoder = ParallelTransformer(
            config,
            model_type=args.model_type if not args.retro_add_retriever \
                else ModelType.retro_decoder,
            self_attn_mask_type=self.encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self._encoder_key = 'encoder'
    else:
        self.encoder = None

    # Decoder (usually set to False, True if part of an encoder-decoder
    # architecture and in decoder-only stage).
    if self.add_decoder:
        self.decoder = ParallelTransformer(
            config,
            model_type=args.model_type,
            layer_type=LayerType.decoder,
            self_attn_mask_type=self.decoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process)
        self._decoder_key = 'decoder'
    else:
        self.decoder = None

    if self.post_process:
        # Pooler.
        if self.add_pooler:
            self.pooler = Pooler(self.hidden_size, self.init_method)
            self._pooler_key = 'pooler'

        if self.untie_embeddings_and_output_weights:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                args.padded_vocab_size,
                config=config,
                init_method=self.init_method,
                bias=False) # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
            self._output_layer_key = 'output_layer'


def TransformerLanguageModelForward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                enc_dec_attn_mask=None, tokentype_ids=None,
                inference_params=None,
                pooling_sequence_index=0,
                enc_hidden_states=None, output_enc_hidden=False,
                past_key_values=None, use_cache=False):
    self.seq_length = get_args().seq_length

    # Encoder embedding.
    if self.pre_process:
        encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                       tokentype_ids=tokentype_ids)
    else:
        encoder_input = None

    # Retriever embedding.
    if self.add_retriever and self.pre_process:
        retriever_input = self.embedding(retriever_input_ids,
                                         retriever_position_ids,
                                         tokentype_ids=tokentype_ids)
    else:
        retriever_input = None

    # Rotary positional embeddings
    rotary_pos_emb = None
    if self.use_rotary_position_embeddings:
        if inference_params is not None:
            rotary_pos_emb = \
                self.rotary_pos_emb(inference_params.max_sequence_length)
        else:
            rotary_pos_emb = self.rotary_pos_emb(self.seq_length)

    # Run encoder.
    if enc_hidden_states is None:
        if self.encoder is not None:
            encoder_output = self.encoder(
                encoder_input,
                enc_attn_mask,
                retriever_input=retriever_input,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                past_key_values=past_key_values,  # Add past_key_values and use_cache for inference
                use_cache=use_cache)
            if use_cache:
                encoder_output, past_key_values = encoder_output
        else:
            encoder_output = self.encoder_hidden_state
    else:
        encoder_output = enc_hidden_states.to(encoder_input.dtype)

    if self.post_process:
        if self.add_pooler:
            pooled_output = self.pooler(encoder_output,
                                        pooling_sequence_index)

    # output_enc_hidden refers to when we just need the encoder's
    # output. For example, it is helpful to compute
    # similarity between two sequences by average pooling
    if not self.add_decoder or output_enc_hidden:
        # Handle output within past_key_values
        if use_cache:
            encoder_output = encoder_output, past_key_values
        if self.add_pooler and self.post_process:
            return encoder_output, pooled_output
        else:
            return encoder_output

    # Decoder embedding.
    if self.pre_process:
        decoder_input = self.embedding(dec_input_ids,
                                       dec_position_ids)
    else:
        decoder_input = None

    # Run decoder.
    decoder_output = self.decoder(
        decoder_input,
        dec_attn_mask,
        encoder_output=encoder_output,
        enc_dec_attn_mask=enc_dec_attn_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb)

    if self.add_pooler and self.post_process:
        return decoder_output, encoder_output, pooled_output
    else:
        return decoder_output, encoder_output
