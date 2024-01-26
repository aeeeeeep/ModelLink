# coding=utf-8
# rewritten, Copyright (c) 2021, Ming Ding.  All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""

import copy
import gc
import json
import math
import os

from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
import torch
import torch.nn.functional as F

from SwissArmyTransformer import mpu
from SwissArmyTransformer.mpu.initialize import get_model_parallel_world_size
from SwissArmyTransformer.mpu.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from SwissArmyTransformer.mpu.mappings import gather_from_model_parallel_region, copy_to_model_parallel_region
from SwissArmyTransformer.mpu.utils import divide, sqrt, scaled_init_method, unscaled_init_method, gelu
from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim
from SwissArmyTransformer.model.position_embedding import RotaryEmbedding
from SwissArmyTransformer.model.position_embedding import apply_rotary_pos_emb_index
from SwissArmyTransformer.ops import LayerNorm
from SwissArmyTransformer.transformer_defaults import HOOKS_DEFAULT, standard_attention


ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
if ATB_SPEED_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(ATB_SPEED_HOME_PATH,
                        "lib/libatb_speed_torch.so")
torch.classes.load_library(LIB_PATH)


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True,
                 hooks={}, transformer_pointer=None, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(
                hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(
            num_attention_heads, world_size)
        self.inner_hidden_size = num_attention_heads * \
            self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * \
            self.num_attention_heads_per_partition

        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(
            hidden_size,
            3 * self.inner_hidden_size,
            stride=3,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="query_key_value",
            skip_init=skip_init,
            device=device
        )
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = RowParallelLinear(
            self.inner_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense",
            skip_init=skip_init,
            device=device
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        object.__setattr__(self, 'transformer', transformer_pointer)
        if transformer_pointer is None:
            raise RuntimeError(f'Invalid transformer_pointer: {transformer_pointer}')

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask, *args, **kw_args):
        if 'attention_forward' in self.hooks:
            return self.hooks['attention_forward'](hidden_states, mask, **kw_args)
        else:
            return HOOKS_DEFAULT['attention_forward'](self, hidden_states, mask, **kw_args)


class CrossAttention(torch.nn.Module):
    """Parallel cross-attention layer for Transformer"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method,
                 layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True, hooks={},
                 transformer_pointer=None, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(
                hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(
            num_attention_heads, world_size)
        self.inner_hidden_size = num_attention_heads * \
            self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * \
            self.num_attention_heads_per_partition
        # Strided linear layer.
        self.query = ColumnParallelLinear(hidden_size, self.inner_hidden_size,
                                          gather_output=False,
                                          init_method=init_method, bias=bias, params_dtype=params_dtype, module=self, name="query", skip_init=skip_init, device=device)
        self.key_value = ColumnParallelLinear(hidden_size, 2 * self.inner_hidden_size,
                                              stride=2,
                                              gather_output=False,
                                              init_method=init_method, bias=bias, params_dtype=params_dtype, module=self, name="key_value",
                                              skip_init=skip_init, device=device)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(
            self.inner_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method, bias=bias, params_dtype=params_dtype, module=self, name="dense", skip_init=skip_init,
            device=device)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        object.__setattr__(self, 'transformer', transformer_pointer)
        if transformer_pointer is None:
            raise RuntimeError(f'Invalid transformer_pointer: {transformer_pointer}')

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
        # hidden_states: [b, s, h]
        if 'cross_attention_forward' in self.hooks:
            return self.hooks['cross_attention_forward'](hidden_states, cross_attention_mask, encoder_outputs, **kw_args)
        else:
            return HOOKS_DEFAULT['cross_attention_forward'](self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args)


class MLP(torch.nn.Module):
    def __init__(self, hidden_size, output_dropout_prob, init_method, inner_hidden_size=None,
                 output_layer_init_method=None, layer_id=None, hooks={}, bias=True, activation_func=gelu, transformer_pointer=None, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super(MLP, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            self.hidden_size,
            self.inner_hidden_size,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense_h_to_4h",
            skip_init=skip_init,
            device=device
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            self.inner_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense_4h_to_h",
            skip_init=skip_init,
            device=device
        )
        self.dropout = torch.nn.Dropout(output_dropout_prob)
        object.__setattr__(self, 'transformer', transformer_pointer)
        if transformer_pointer is None:
            raise RuntimeError(f'Invalid transformer_pointer: {transformer_pointer}')

    def forward(self, hidden_states, **kw_args):
        if 'mlp_forward' in self.hooks:
            output = self.hooks['mlp_forward'](hidden_states, **kw_args)
        else:
            output = HOOKS_DEFAULT['mlp_forward'](
                self, hidden_states, **kw_args)

        if self.training:
            output = self.dropout(output)
        return output


class BaseTransformerLayer(torch.nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            layernorm_epsilon,
            init_method,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            output_layer_init_method=None,
            layernorm_order='pre',
            layernorm=LayerNorm,
            is_decoder=False,
            use_bias=True,
            activation_func=gelu,
            hooks={},
            transformer_pointer=None,
            params_dtype=torch.float,
            skip_init=False,
            device=torch.device('cpu')
    ):
        super(BaseTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.is_decoder = is_decoder
        self.layernorm_order = layernorm_order
        self.hooks = hooks
        object.__setattr__(self, 'transformer', transformer_pointer)
        if transformer_pointer is None:
            raise RuntimeError(f'Invalid transformer_pointer: {transformer_pointer}')

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(
            hidden_size, eps=layernorm_epsilon)
        if self.layernorm_order == 'sandwich':
            self.third_layernorm = layernorm(
                hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = layernorm(
                hidden_size, eps=layernorm_epsilon)

        # Cross attention.
        if self.is_decoder:
            self.cross_attention = CrossAttention(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                init_method,
                layer_id,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=output_layer_init_method,
                bias=use_bias,
                hooks=hooks,
                transformer_pointer=transformer_pointer,
                params_dtype=params_dtype
            )
            self.post_cross_attention_layernorm = layernorm(
                hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            inner_hidden_size=inner_hidden_size,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            layer_id=layer_id,
            activation_func=activation_func,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device
        )

    def forward(self, hidden_states, mask, *args, **kw_args):
        return HOOKS_DEFAULT['layer_forward'](self, hidden_states, mask, *args, **kw_args)


class BaseTransformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 inner_hidden_size=None,
                 hidden_size_per_attention_head=None,
                 layernorm_order='pre',
                 parallel_output=True,
                 is_decoder=False,
                 use_bias=True,
                 activation_func=gelu,
                 layernorm=LayerNorm,
                 init_method=None,
                 use_final_layernorm=True,
                 hooks={},
                 params_dtype=torch.float,
                 skip_init=False,
                 device=torch.device('cpu'),
                 atb_backend='hccl'
                 ):
        super(BaseTransformer, self).__init__()

        # recording parameters
        self.is_decoder = is_decoder
        self.parallel_output = parallel_output
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_sequence_length = max_sequence_length
        self.layernorm_order = layernorm_order
        self.hooks = copy.copy(hooks)  # hooks will be updated each forward
        # to give the default hooks the same api as outer hooks
        object.__setattr__(self, 'transformer', self)

        # create embedding parameters
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size,
            params_dtype=params_dtype, skip_init=skip_init, device=device)

        self.position_embeddings = torch.nn.Embedding(
            max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight,
                              mean=0.0, std=init_method_std)

        # create all layers
        if init_method is None:
            self.output_layer_init_method = scaled_init_method(
                init_method_std, num_layers)
            self.init_method = unscaled_init_method(init_method_std)
        else:
            self.output_layer_init_method = init_method
            self.init_method = init_method

        def get_layer(layer_id):
            return BaseTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                self.init_method,
                layer_id,
                inner_hidden_size=inner_hidden_size,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=self.output_layer_init_method,
                is_decoder=self.is_decoder,
                layernorm_order=layernorm_order,
                layernorm=layernorm,
                use_bias=use_bias,
                activation_func=activation_func,
                hooks=self.hooks,
                transformer_pointer=self,
                params_dtype=params_dtype,
                skip_init=skip_init,
                device=device
            )

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)])

        # Final layer norm before output.
        self.use_final_layernorm = use_final_layernorm
        if use_final_layernorm:
            self.final_layernorm = layernorm(
                hidden_size, eps=layernorm_epsilon)

        # ATB code---------------------------------------------------------------
        self.rankSize = get_model_parallel_world_size()
        self.rank = torch.distributed.get_rank()
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_attention_heads
        self.headNum = num_attention_heads
        self.num_layers = num_layers
        self.rotary_emb = RotaryEmbedding(
            self.head_size,
            base=10000,
            precision=torch.half,
            learnable=False,
            device=torch.cuda.current_device(),
        )
        self.cos_table = None
        self.sin_table = None
        self.weight_flag = True

        # flash attention init
        self.batch_num = 1  # 默认batch_size=1
        self.layernorm_epsilon = layernorm_epsilon

        self.seq_lens = None
        self.qScale = [(1 / (math.sqrt(self.head_size) * float(layer_id + 1)))
                       for layer_id in range(self.num_layers)]
        self.qkScale = [float(layer_id + 1)
                        for layer_id in range(self.num_layers)]
        self.layer_indexes = [
            torch.tensor([i], dtype=torch.int32, device=torch.cuda.current_device())
            for i in range(self.num_layers)
        ]

        atb_param = {
            "layerNum": self.num_layers,
            "headNum": self.headNum,
            "headDim": self.head_size,
            "qScale": self.qScale,
            "qkScale": self.qkScale,
            "rank": self.rank,
            "rankSize": self.rankSize,
            "rankRoot": 0,
            "backend": atb_backend,
            "residualAddScale": (2 * num_layers) ** 0.5,
            "layerNormEps": self.layernorm_epsilon
        }
        atb_decoder_param = atb_param.copy()
        atb_decoder_param.update({"coderType": 2})
        atb_encoder_param = atb_param.copy()
        atb_encoder_param.update({"coderType": 1})
        self.atb_decoder_operation = torch.classes.ModelTorch.ModelTorch(
            "glm_130b_fusion_parallel_model")
        self.atb_decoder_operation.set_param(json.dumps(atb_decoder_param))
        self.atb_encoder_operation = torch.classes.ModelTorch.ModelTorch(
            "glm_130b_fusion_parallel_model")
        self.atb_encoder_operation.set_param(json.dumps(atb_encoder_param))
        # ATB code---------------------------------------------------------------

    def forward(self, input_ids, position_ids, attention_mask, *,
                output_hidden_states=False, **kw_args):
        if self.checkpoint_activations:
            raise RuntimeError("only support inference mode!")
        # sanity check
        if len(input_ids.shape) < 2:
            raise RuntimeError("Invalid input_ids shape:", input_ids.shape)

        batch_size, query_length = input_ids.shape[:2]

        if not hasattr(self, "kv_cache") or batch_size != self.batch_num:
            self.batch_num = batch_size
            self.kv_cache = torch.zeros(2,
                                        self.num_layers,
                                        self.batch_num,   # batch
                                        self.max_sequence_length,
                                        self.headNum // self.rankSize,
                                        self.head_size,
                                        device=torch.cuda.current_device(), dtype=torch.half).contiguous()
            self.k_cache_input = self.kv_cache[0].view(
                self.num_layers, self.batch_num, self.max_sequence_length, self.hidden_size // self.rankSize)
            self.v_cache_input = self.kv_cache[1].view(
                self.num_layers, self.batch_num, self.max_sequence_length, self.hidden_size // self.rankSize)
            
            self.tokens_offset = torch.full(
                (self.batch_num,), 0, dtype=torch.int32, device=self.kv_cache.device)
            self.attention_mask_max = torch.full(
                (self.batch_num, self.max_sequence_length, self.max_sequence_length), 
                1, dtype=torch.half, device=self.kv_cache.device)

        # initial output_cross_layer might be generated by word/position_embedding_forward
        output_cross_layer = {}

        # embedding part
        if 'word_embedding_forward' in self.hooks:
            hidden_states = self.hooks['word_embedding_forward'](
                input_ids, output_cross_layer=output_cross_layer, **kw_args)
        else:  # default
            hidden_states = HOOKS_DEFAULT['word_embedding_forward'](
                self, input_ids, output_cross_layer=output_cross_layer, **kw_args)

        # atb code
        if self.weight_flag:
            atb_weights = []
            for layer in self.layers:
                atb_layer_weights = list(layer.state_dict().values())
                atb_weights.extend(atb_layer_weights[0:8])
                atb_weights.extend(atb_layer_weights[10:12])
                atb_weights.extend(atb_layer_weights[8:10])

            atb_model_weights = list(self.state_dict().values())
            atb_weights.append(
                atb_model_weights[-3])  # final norm weight
            atb_weights.append(
                atb_model_weights[-2])  # final norm bias
            atb_weights.append(
                atb_model_weights[0])  # final forward weight

            self.atb_decoder_operation.set_weight(atb_weights)
            self.atb_encoder_operation.set_weight(atb_weights)
            self.weight_flag = False

            self.cos_table, self.sin_table = self.rotary_emb(
                hidden_states, seq_len=self.max_sequence_length + 1)

            del atb_weights
            gc.collect()
            torch.npu.empty_cache()

        logits_atb = None
        output_per_layers = []
        operation = None

        # full
        if query_length > 1:
            bos_token_id = 150004
            context_lengths = [seq.tolist().index(bos_token_id)
                               for seq in input_ids]  # 多batch时，input_ids会进行padding
            self.seq_lens = torch.tensor(
                [query_length] * batch_size, device=self.kv_cache.device, dtype=torch.int32)
            self.tokens_offset = torch.tensor(
                [query_length] * batch_size, device=self.kv_cache.device, dtype=torch.int32)
            
            self.attention_mask_max.triu_(diagonal=1)
            for i, context_length in enumerate(context_lengths):
                self.attention_mask_max[i, :, :context_length] = 0
            self.attention_mask_max *= -10000.0

            operation = self.atb_encoder_operation
        else:
            operation = self.atb_decoder_operation


        atb_param = json.dumps({
            "tokenOffset": self.tokens_offset.tolist(),
            "seqLen": self.seq_lens.tolist()
        })
        atb_model_inputs = [
            hidden_states.transpose(0, 1),  # change to [bs,seq_len,...]
            position_ids,
            self.cos_table.squeeze(1),
            self.sin_table.squeeze(1),
            self.attention_mask_max,
            self.k_cache_input,
            self.v_cache_input,
            self.tokens_offset,
            self.seq_lens,
        ]
        atb_model_out = operation.execute(atb_model_inputs + self.layer_indexes, atb_param)
        logits_atb = atb_model_out[0]
        self.tokens_offset.add_(1)

        if query_length > 1:
            self.seq_lens.fill_(1)
            self.attention_mask_max.fill_(0)

        outputs = [logits_atb]
        outputs.extend(output_per_layers)

        # Large sequence length easily cause a OOM(out of memory) error, to avoid it,
        # we can find a critical point where OOM occurs, when sequence length greater
        # than the point, we need to deallocate memory of intermediate variables. Of
        # course, there may be a little decrease in performance.
        if query_length > 1536:
            del atb_model_inputs
            del self.kv_cache
            del self.k_cache_input
            del self.v_cache_input
            del self.tokens_offset
            del self.attention_mask_max
            torch.npu.empty_cache()
    
        return outputs
