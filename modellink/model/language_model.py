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

from functools import wraps

from megatron import get_args
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import TransformerLanguageModel, Embedding
from megatron.model.transformer import ParallelTransformer
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