# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch TELECHAT model."""

import math
import copy
import warnings
from typing import Optional, Tuple, Union, List, Dict
from threading import Thread

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from apex.normalization import FusedLayerNorm
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_telechat import TelechatConfig
from .generation_utils import History, TelechatIterTextStreamer
from transformers import GenerationConfig

# add acceration
import os
import json
import numpy as np
import torch_npu

ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
if ATB_SPEED_HOME_PATH is None:
        raise RuntimeError("env ATB_SPEED_HOME_PATH  not exist, source set_env.sh")
LIB_PATH = os.path.join(ATB_SPEED_HOME_PATH ,"lib/libatb_speed_torch.so")
torch.classes.load_library(LIB_PATH)

RUN_QUANT_MODEL = os.environ.get("RUN_QUANT_MODEL", 0)
QUANT_WEIGHT_PATH = os.environ.get("QUANT_WEIGHT_PATH", "")
MAX_SEQ_LEN = os.environ.get("MAX_SEQ_LEN", 2048)
FLOAT_QUERY_LAYERS = []
FLOAT_KV_LAYERS = []
FLOAT_DOWN_LAYERS = [0,1,9,25,27]

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigscience/telechat-560m"
_CONFIG_FOR_DOC = "TelechatConfig"

TELECHAT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/telechat-560m",
    "bigscience/telechat-1b1",
    "bigscience/telechat-1b7",
    "bigscience/telechat-3b",
    "bigscience/telechat-7b1",
    "bigscience/telechat",
]
lm_head_weight = None


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.full((target_length, target_length), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    #seq_ids = torch.arange(target_length, device=device)
    #mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(target_length, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        # mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    #expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_length, src_length).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class TelechatRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        if RUN_QUANT_MODEL:
            self.bias = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        if RUN_QUANT_MODEL:
            return self.weight * hidden_states + self.bias
        else:
            return self.weight * hidden_states

class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        # inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().half() / dim))
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().double() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=0, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def telechat_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def telechat_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)

    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return telechat_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = telechat_gelu_back(grad_output, input)
        return tmp


class TelechatGelu(nn.Module):
    """
    TelechatBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return telechat_gelu_forward(x)


class TelechatAttention(nn.Module):
    def __init__(self, config: TelechatConfig,layer_idx):
        super().__init__()
        self.world_size = 1
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        self.kv_cache = None
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.num_key_value_heads = 32
        kv_projection_size = self.head_dim * self.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size // self.world_size, bias=False)
        self.key_value = nn.Linear(self.hidden_size, kv_projection_size * 2 // self.world_size, bias=False)
        self.dense = nn.Linear(self.hidden_size // self.world_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.num_heads = self.num_heads // self.world_size        
        self.num_key_value_heads = self.num_key_value_heads // self.world_size

        self.last_key_layer = None
        
    def forward(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            alibi: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            head_mask: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        return


class TelechatMLP(nn.Module):
    def __init__(self, config: TelechatConfig, layer_idx):
        super().__init__()
        hidden_size = config.hidden_size
        self.world_size = 1
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        if not RUN_QUANT_MODEL:
            self.gate_proj = nn.Linear(hidden_size, config.ffn_hidden_size // self.world_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, config.ffn_hidden_size // self.world_size, bias=False)
        if not RUN_QUANT_MODEL or layer_idx in FLOAT_DOWN_LAYERS:
            self.down_proj = nn.Linear(config.ffn_hidden_size // self.world_size, hidden_size, bias=True)

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)  ###split x into 2 tensors along the last dimension

        return F.silu(x[0]) * x[1]

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return


class TelechatBlock(nn.Module):
    def __init__(self, config: TelechatConfig,layer_idx):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = TelechatRMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.world_size = 1
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size

        self.num_heads = config.n_head // self.world_size
        self.layer_idx = layer_idx
        if RUN_QUANT_MODEL:
            self.self_attention = TelechatAttention(config, layer_idx)
        self.post_attention_layernorm = TelechatRMSNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = TelechatMLP(config, layer_idx)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
            self,
            hidden_states: torch.Tensor,
            alibi: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            head_mask: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        return


class TelechatPreTrainedModel(PreTrainedModel):
    config_class = TelechatConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TelechatBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False):
        if isinstance(module, TelechatModel):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_to_standard_cache(
            past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_telechat_cache(
            past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Telechat, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


TELECHAT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TelechatConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TELECHAT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]`
            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.

            Each element of `past_key_values` is a tuple (past_key, past_value):
            - past_key: [batch_size * num_heads, head_dim, kv_length]
            - past_value: [batch_size * num_heads, kv_length, head_dim]
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare Telechat Model transformer outputting raw hidden-states without any specific head on top.",
    TELECHAT_START_DOCSTRING,
)

def process_deq_scale(deq_scale_dict):
    new_deq_scale_dict = {}
    for key, deq_scale in deq_scale_dict.items():
        new_deq_scale = np.frombuffer(deq_scale.numpy().tobytes(), dtype=np.int32)
        new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
    return new_deq_scale_dict

def bias_correction_new(fp_bias, quant_weight, input_offset, deq_scale):
    bias_correction = fp_bias.npu() / deq_scale.npu() - quant_weight.to(torch.float32).npu().sum(dim=1) * float(input_offset)
    return bias_correction

def get_bias_correction_dict(fp_bias_dict, quant_weight_dict, input_offset_dict, deq_scale_dict):
    bias_dict = {}
    for k in fp_bias_dict.keys():
        bias_dict[k] = bias_correction_new(fp_bias_dict[k], quant_weight_dict[k], input_offset_dict[k], deq_scale_dict[k])
    return bias_dict

class TelechatModel(TelechatPreTrainedModel):
    def __init__(self, config: TelechatConfig):
        super().__init__(config)
        print("start TelechatModel")
        self.world_size = 1
        self.rank = 0
        self.rankSize = 1

        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
            self.rankSize = self.world_size
            self.rank = torch.distributed.get_rank()

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([TelechatBlock(config,_) for _ in range(config.num_hidden_layers)])
        # Final Layer Norm
        self.ln_f = TelechatRMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self.dk = self.embed_dim // self.num_heads
        self.num_heads = self.num_heads // self.world_size
        # add  model op
        self.num_hidden_layers = config.num_hidden_layers

        # Initialize weigts and apply final processing
        self.post_init()
        
        print("self.rank", self.rank, "self.rankSize", self.rankSize)
        print("current_device", torch_npu.npu.current_device())
        print("device count", torch.npu.device_count())

        # rotary pos emb helpers:
        zero_tensor = torch.zeros(1)
        rot_emb_global = RotaryEmbedding(128)
        cosTable, sinTable = rot_emb_global.forward(zero_tensor, seq_len=MAX_SEQ_LEN)
        self.cosTable = cosTable.npu().half()
        self.sinTable = sinTable.npu().half()
        
        # QUANT
        if RUN_QUANT_MODEL:
            self.float_query_layers = FLOAT_QUERY_LAYERS
            self.float_kv_layers = FLOAT_KV_LAYERS
            self.float_down_layers = FLOAT_DOWN_LAYERS
            if self.world_size > 1:
                quant_param_path = os.path.join(QUANT_WEIGHT_PATH, "part_model", str(self.rank))
            else:
                quant_param_path = QUANT_WEIGHT_PATH
            self.input_scale_dict = np.load(os.path.join(quant_param_path, "input_scale.npy"), allow_pickle=True).item()
            self.input_offset_dict = np.load(os.path.join(quant_param_path, "input_offset.npy"), allow_pickle=True).item()
            self.quant_weight_dict = np.load(os.path.join(quant_param_path, "quant_weight.npy"), allow_pickle=True).item()
            self.weight_scale_dict = np.load(os.path.join(quant_param_path, "weight_scale.npy"), allow_pickle=True).item()
            self.deq_scale_dict = np.load(os.path.join(quant_param_path, "deq_scale.npy"), allow_pickle=True).item()
            self.fp_bias_dict = np.load(os.path.join(quant_param_path, "fp_bias.npy"), allow_pickle=True).item()
            if self.world_size == 1:
                self.fp_bias_dict = get_bias_correction_dict(self.fp_bias_dict, self.quant_weight_dict, self.input_offset_dict, self.deq_scale_dict)
                self.deq_scale_dict = process_deq_scale(self.deq_scale_dict)
        
        zero_list = [ 0 for _ in range(self.num_hidden_layers)]
        qkv_input_scale = zero_list[:]
        qkv_input_offset = zero_list[:]
        dense_input_scale = zero_list[:]
        dense_input_offset = zero_list[:]
        gate_up_input_scale = zero_list[:]
        gate_up_input_offset = zero_list[:]
        down_proj_input_scale = zero_list[:]
        down_proj_input_offset = zero_list[:]

        self.bias_list = []
        self.weights = []
        for layer_idx in range(self.num_hidden_layers):
            query_name = "transformer.h.{}.self_attention.query".format(layer_idx)
            key_value_name = "transformer.h.{}.self_attention.key_value".format(layer_idx)
            dense_name = "transformer.h.{}.self_attention.dense".format(layer_idx)
            gate_proj_name = "transformer.h.{}.mlp.gate_proj".format(layer_idx)
            up_proj_name = "transformer.h.{}.mlp.up_proj".format(layer_idx)
            down_proj_name = "transformer.h.{}.mlp.down_proj".format(layer_idx)

            if RUN_QUANT_MODEL:
                dense_input_scale[layer_idx] = float(1 / self.input_scale_dict[dense_name])
                dense_input_offset[layer_idx] = int(self.input_offset_dict[dense_name])
                gate_up_input_scale[layer_idx] = float(1 / self.input_scale_dict[gate_proj_name])
                gate_up_input_offset[layer_idx] = int(self.input_offset_dict[gate_proj_name])

            if layer_idx not in self.float_query_layers:
                qkv_input_scale[layer_idx] = float(1 / self.input_scale_dict[query_name])
                qkv_input_offset[layer_idx] = int(self.input_offset_dict[query_name])
                query_quant_bias = self.fp_bias_dict[query_name]
            else:
                query_quant_bias = [0]

            if layer_idx not in self.float_down_layers:
                down_proj_input_scale[layer_idx] = float(1 / self.input_scale_dict[down_proj_name])
                down_proj_input_offset[layer_idx] = int(self.input_offset_dict[down_proj_name])
                down_proj_quant_bias = self.fp_bias_dict[down_proj_name]
            else:
                down_proj_quant_bias = [0]

            if layer_idx in self.float_kv_layers:
                key_value_quant_bias = [0]
            else:
                key_value_quant_bias = self.fp_bias_dict[key_value_name]

            dense_quant_bias = self.fp_bias_dict[dense_name]
            gate_proj_quant_bias = self.fp_bias_dict[gate_proj_name]
            up_proj_quant_bias = self.fp_bias_dict[up_proj_name]

            bias_pre_layer = [
                       query_quant_bias,
                       key_value_quant_bias,
                       dense_quant_bias,
                       gate_proj_quant_bias,
                       up_proj_quant_bias,
                       down_proj_quant_bias
                       ]
            self.bias_list.append(bias_pre_layer)

        self.acl_param = json.dumps({
            "dk": self.dk,
            "headNum": self.num_heads,
            "float_query_layers":self.float_query_layers,
            "float_kv_layers":self.float_kv_layers,
            "float_down_layers":self.float_down_layers,
            "rmsNormEps":config.layer_norm_epsilon,
            "rank": self.rank,
            "rankSize": self.rankSize,
            "layerNum": self.num_hidden_layers,
            "inputScale_qkv": qkv_input_scale,
            "inputOffset_qkv": qkv_input_offset,
            "inputScale_dense": dense_input_scale,
            "inputOffset_dense": dense_input_offset,
            "inputScale_gate_up": gate_up_input_scale,
            "inputOffset_gate_up": gate_up_input_offset,
            "inputScale_down_proj": down_proj_input_scale,
            "inputOffset_down_proj": down_proj_input_offset,
            })

        if RUN_QUANT_MODEL:
            self.acl_operation = torch.classes.ModelTorch.ModelTorch("telechat_quant_model")
        else:
            self.acl_operation = torch.classes.ModelTorch.ModelTorch("telechat_float_model")

        print("set param")
        self.acl_operation.set_param(self.acl_param)
        self.weightFlag=False
        self.encoder_flag = True
        num_intensors = 10
        self.acl_inputs = [None] * (num_intensors + self.num_hidden_layers)
        for i in range(self.num_hidden_layers):
            self.acl_inputs[i + num_intensors] = torch.tensor([i],dtype=torch.int32).npu()

        self.seq_len = 0
        self.max_seq_len = MAX_SEQ_LEN
        self.batch_num = 0
        self.cached_k = None
        self.cached_v = None

        self.maskAttenfull = None
        self.maskAttenincre = None
        self.maskAttenincreCache = torch.full((self.max_seq_len, self.max_seq_len), torch.finfo(torch.float16).min, device='npu', dtype=torch.half)
        self.maskAttenincreZero = torch.full((self.max_seq_len, self.max_seq_len), 0, device='npu', dtype=torch.half)

        #transfer quant weight from ND to NZ
        transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        transdata_operation.set_param(transdata_param)

        self.model_weights = [[] for i in range(self.num_hidden_layers)]
        for layer_idx in range(self.num_hidden_layers):
            query_name = "transformer.h.{}.self_attention.query".format(layer_idx)
            key_value_name = "transformer.h.{}.self_attention.key_value".format(layer_idx)
            dense_name = "transformer.h.{}.self_attention.dense".format(layer_idx)
            gate_proj_name = "transformer.h.{}.mlp.gate_proj".format(layer_idx)
            up_proj_name = "transformer.h.{}.mlp.up_proj".format(layer_idx)
            down_proj_name = "transformer.h.{}.mlp.down_proj".format(layer_idx)
            weights_float = self.h[layer_idx].state_dict()
           

            if layer_idx in self.float_query_layers:
                self.model_weights[layer_idx].append(weights_float["self_attention.query.weight"].npu())
            else:
                self.model_weights[layer_idx].append(transdata_operation.execute([self.quant_weight_dict[query_name].to(torch.int8).npu()])[0])
            if layer_idx in self.float_kv_layers:
                self.model_weights[layer_idx].append(weights_float["self_attention.key_value.weight"].npu())
            else:
                self.model_weights[layer_idx].append(transdata_operation.execute([self.quant_weight_dict[key_value_name].to(torch.int8).npu()])[0])

            self.model_weights[layer_idx].append(transdata_operation.execute([self.quant_weight_dict[dense_name].to(torch.int8).npu()])[0])
            self.model_weights[layer_idx].append(transdata_operation.execute([self.quant_weight_dict[gate_proj_name].to(torch.int8).npu()])[0])
            self.model_weights[layer_idx].append(transdata_operation.execute([self.quant_weight_dict[up_proj_name].to(torch.int8).npu()])[0])

            if layer_idx in self.float_down_layers:
                self.model_weights[layer_idx].append(weights_float['mlp.down_proj.weight'].npu())
            else:
                self.model_weights[layer_idx].append(transdata_operation.execute([self.quant_weight_dict[down_proj_name].to(torch.int8).npu()])[0])
        print("end TelechatModel")

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
            self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, torch.float16, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, torch.float16, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(TELECHAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        # print(f"self.state_dict key is {self.state_dict().keys()}")
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in TELECHAT and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        if seq_length > 1:
            self.encoder_flag = True

        if self.batch_num != batch_size:
            self.batch_num = batch_size
            self.hidden_size_nz = self.embed_dim // self.world_size // 16
            self.cached_k = torch.zeros(self.num_hidden_layers, self.batch_num,  self.hidden_size_nz, self.max_seq_len, 16, device = "npu").half().contiguous()
            self.cached_v = torch.zeros(self.num_hidden_layers, self.batch_num,  self.hidden_size_nz, self.max_seq_len, 16, device = "npu").half().contiguous()
            self.cached_k.data = torch_npu.npu_format_cast(self.cached_k.data, 29)
            self.cached_v.data = torch_npu.npu_format_cast(self.cached_v.data, 29)

        if self.encoder_flag:
            self.maskAttenfull = torch.full((self.batch_num, self.max_seq_len, self.max_seq_len), 0, device='npu', dtype=torch.half)
            self.maskAttenincre = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if not self.encoder_flag:
            past_key_values_length = self.seq_len
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=input_ids.device)
        else:
            attention_mask = attention_mask.to(input_ids.device)
        alibi = None
        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )
        if self.weightFlag is False:

            global lm_head_weight
            self.weights.append(self.state_dict()["word_embeddings.weight"].npu())
            zero_tensor = torch.tensor([0]).npu()
                                            
            for layer_idx in range(self.num_hidden_layers):
                query_name = "transformer.h.{}.self_attention.query".format(layer_idx)
                key_value_name = "transformer.h.{}.self_attention.key_value".format(layer_idx)
                dense_name = "transformer.h.{}.self_attention.dense".format(layer_idx)
                gate_proj_name = "transformer.h.{}.mlp.gate_proj".format(layer_idx)
                up_proj_name = "transformer.h.{}.mlp.up_proj".format(layer_idx)
                down_proj_name = "transformer.h.{}.mlp.down_proj".format(layer_idx)
                weights_all = []
                weights_float = self.h[layer_idx].state_dict()
                if layer_idx in self.float_query_layers:
                    weights_q = [
                                self.model_weights[layer_idx][0],
                                zero_tensor,
                                zero_tensor,
                                ]
                else:
                    weights_q = [
                                self.model_weights[layer_idx][0].to(torch.int8),
                                self.deq_scale_dict[query_name].to(torch.int64).npu(),
                                self.bias_list[layer_idx][0].to(torch.int32).npu()]
                weights_all.extend(weights_q)
                if layer_idx in self.float_kv_layers:
                    weights_kv = [
                                self.model_weights[layer_idx][1],
                                zero_tensor,
                                zero_tensor,
                                ]
                else:
                    weights_kv = [
                                self.model_weights[layer_idx][1].to(torch.int8),
                                self.deq_scale_dict[key_value_name].to(torch.int64).npu(),
                                self.bias_list[layer_idx][1].to(torch.int32).npu()]
                weights_all.extend(weights_kv)
                weights_dense = [
                                self.model_weights[layer_idx][2].to(torch.int8),
                                self.deq_scale_dict[dense_name].to(torch.int64).npu(),
                                self.bias_list[layer_idx][2].to(torch.int32).npu()]
                weights_all.extend(weights_dense)
                weights_gate = [
                                self.model_weights[layer_idx][3].to(torch.int8),
                                self.deq_scale_dict[gate_proj_name].to(torch.int64).npu(),
                                self.bias_list[layer_idx][3].to(torch.int32).npu()]
                weights_all.extend(weights_gate)
                weights_up = [
                            self.model_weights[layer_idx][4].to(torch.int8),
                            self.deq_scale_dict[up_proj_name].to(torch.int64).npu(),
                            self.bias_list[layer_idx][4].to(torch.int32).npu(),
                            ]
                weights_all.extend(weights_up)
                if layer_idx in self.float_down_layers:
                    weights_down = [
                                    weights_float['mlp.down_proj.weight'],
                                    zero_tensor,
                                    weights_float['mlp.down_proj.bias'].npu()
                                    ]
                else:
                    weights_down = [
                                    self.model_weights[layer_idx][5].to(torch.int8),
                                    self.deq_scale_dict[down_proj_name].to(torch.int64).npu(),
                                    self.bias_list[layer_idx][5].to(torch.int32).npu()
                                    ]
                weights_all.extend(weights_down)
                self.weights.extend(weights_all)
                self.weights.append(weights_float['input_layernorm.weight'].npu())
                self.weights.append(weights_float['input_layernorm.bias'].npu())
                self.weights.append(weights_float['post_attention_layernorm.weight'].npu())
                self.weights.append(weights_float['post_attention_layernorm.bias'].npu())

            self.weights.append(self.state_dict()["ln_f.weight"].npu())
            self.weights.append(lm_head_weight)
            self.acl_operation.set_weight(self.weights)
            self.weightFlag = True

        self.acl_inputs[0] = input_ids
        if self.encoder_flag:
            if self.batch_num == 1:
                self.maskAttenfull[:, :seq_length, :seq_length] = causal_mask[0]
                count = torch.eq(causal_mask[0][0][seq_length - 1], 0).sum().item()
                decoder_leftmask = self.maskAttenincreCache[:, :seq_length - count]
                decoder_rightmask = self.maskAttenincreZero[:, :self.max_seq_len - seq_length + count]
                self.maskAttenincre = torch.concat([decoder_leftmask, decoder_rightmask], dim=-1).unsqueeze(0)
            else:
                decoder_masks = []
                for i in range(self.batch_num):
                    self.maskAttenfull[i][:seq_length, :seq_length] = causal_mask[i][0]
                    count = torch.eq(causal_mask[i][0][seq_length - 1], 0).sum().item()
                    decoder_leftmask = self.maskAttenincreCache[:, :seq_length - count]
                    decoder_rightmask = self.maskAttenincreZero[:, :self.max_seq_len - seq_length + count]
                    decoder_mask = torch.concat([decoder_leftmask, decoder_rightmask], dim=-1).unsqueeze(0)
                    decoder_masks.append(decoder_mask)
                self.maskAttenincre = torch.concat(decoder_masks, dim=0)
            self.maskAttenfull = torch_npu.npu_format_cast(self.maskAttenfull.view(self.batch_num, self.max_seq_len, self.max_seq_len//16, 16).transpose(1,2).contiguous(),29)
            self.maskAttenincre = torch_npu.npu_format_cast(self.maskAttenincre.view(self.batch_num, self.max_seq_len, self.max_seq_len//16, 16).transpose(1,2).contiguous(),29)
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
            position_ids = torch.concat([position_ids] * self.batch_num, dim=0)
            self.acl_inputs[1] = position_ids
            self.acl_inputs[2] = self.cosTable
            self.acl_inputs[3] = self.sinTable
            self.acl_inputs[4] = self.maskAttenfull
            self.acl_inputs[5] = self.cached_k
            self.acl_inputs[6] = self.cached_v
            self.seq_len = input_ids.shape[1]
            seqlen = torch.tensor([self.seq_len] * batch_size, dtype=torch.int32, device=input_ids.device)
            token_offset = torch.tensor([self.seq_len] * batch_size, dtype = torch.int32, device=input_ids.device)
            self.acl_inputs[7] = token_offset
            self.acl_inputs[8] = seqlen
            placeholder = torch.ones(1).npu()
            self.acl_inputs[9] = placeholder
            self.encoder_flag = False
        else:
            offset = self.seq_len
            position_ids = torch.arange(offset, offset + input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
            position_ids = torch.concat([position_ids] * self.batch_num, dim=0)
            self.acl_inputs[1] = position_ids
            self.acl_inputs[2] = self.cosTable
            self.acl_inputs[3] = self.sinTable
            self.acl_inputs[4] = self.maskAttenincre
            self.acl_inputs[5] = self.cached_k
            self.acl_inputs[6] = self.cached_v
            self.seq_len += 1
            seqlen = torch.tensor([1] * batch_size, dtype=torch.int32, device=input_ids.device)
            token_offset = torch.tensor([self.seq_len] * batch_size, dtype = torch.int32, device=input_ids.device)
            self.acl_inputs[7] = token_offset
            self.acl_inputs[8] = seqlen
        param = json.dumps({"tokenOffset":token_offset.tolist(), "seqLen":seqlen.tolist()})
        acl_outputs = self.acl_operation.execute(self.acl_inputs, param)
        hidden_states = acl_outputs[0]
        presents = ((None, None),)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def chat(self, tokenizer, question: str = '', history: Union[List[Dict], History] = None, stream: bool = False,
             generation_config: Optional[GenerationConfig] = None, **kwargs):
        """
        Args:
            tokenizer:  the tokenizer of  telechat
            question: question which the model reply in this turn
            history: history which will format the input for telechat
            stream: if return the full text at last or yield the text in token
            generation_config:  configuration for generation
            **kwargs: args which will update the generation config or pass to model forward
        """
        generation_config = generation_config or self.generation_config
        if not generation_config:
            logger.error("generation_config is None")
            raise ValueError("generation_config must not be None")
        if not question:
            logger.error("question is empty")
            raise ValueError("question must not be empty")
        if history is None:
            history = []

        # we update and check generate_config here for building inputs.

        generation_config = copy.deepcopy(generation_config)
        user_id = generation_config.user_token_id
        bot_id = generation_config.bot_token_id
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()

        # transfer to History
        if not isinstance(history, History):
            history = History(tokenizer, history)

        inputs = self.build_inputs_for_chat(tokenizer, question, history, generation_config, user_id, bot_id)
        history.append({"role": "user", "content": question})
        if stream:
            streamer = TelechatIterTextStreamer(tokenizer, history,skip_prompt=True)
            Thread(target=self.generate, kwargs=dict(
                inputs=inputs.to(self.device), streamer=streamer,
                generation_config=generation_config, **model_kwargs
            )).start()
            return streamer
        else:
            outputs = self.generate(inputs.to(self.device), generation_config=generation_config, **model_kwargs)
            response = tokenizer.decode(outputs[0][len(inputs[0]):-1])
            history.append({"role": "bot", "content": response})
            return response, history

    def build_inputs_for_chat(self, tokenizer, question, history, generation_config, usr_id, bot_id):
        """
        check history and  build inputs here
        """
        # first tokenize question
        q_token = tokenizer(question)
        qa_history = copy.deepcopy(history)

        # get the max length we should build our inputs in
        model_max_length = self.config.seq_length
        build_max_length = max(0, model_max_length - generation_config.max_new_tokens) \
            if generation_config.max_new_tokens else max(0, generation_config.max_length)
        if build_max_length < 3:
            logger.warning("the model can not meet the  requirements of input length,Please check config")
            raise ValueError("")

        # trunc left
        input_tokens = [usr_id] + q_token["input_ids"][-build_max_length + 1:] + [bot_id]
        length = len(input_tokens)

        while len(qa_history) != 0:
            message = qa_history.pop()
            if message["role"] == "user":
                tokens = [usr_id] + message["input_ids"]
            elif message["role"] == "bot":
                tokens = [bot_id] + message["input_ids"] + [generation_config.eos_token_id]
            else:
                tokens = []
            if len(tokens) + length >= build_max_length:
                break
            else:
                input_tokens = tokens + input_tokens

        return torch.tensor([input_tokens], dtype=torch.int64)


@add_start_docstrings(
    """
    The Telechat Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    TELECHAT_START_DOCSTRING,
)
class TelechatForCausalLM(TelechatPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TelechatConfig):
        super().__init__(config)
        self.transformer = TelechatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None

        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            ''' ##changed by ZihanWang
            # the cache may be in the stardard format (e.g. in contrastive search), convert to telechat's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_telechat_cache(past_key_values)
            '''
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(TELECHAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        global lm_head_weight
        if lm_head_weight is None:
            lm_head_weight = self.state_dict()["lm_head.weight"]
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in TELECHAT and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = hidden_states

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
            self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_telechat_cache(reordered_past)


@add_start_docstrings(
    """
    The Telechat Model transformer with a sequence classification head on top (linear layer).

    [`TelechatForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    TELECHAT_START_DOCSTRING,
)
class TelechatForSequenceClassification(TelechatPreTrainedModel):
    def __init__(self, config: TelechatConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = TelechatModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TELECHAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in TELECHAT and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    Telechat Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    TELECHAT_START_DOCSTRING,
)
class TelechatForTokenClassification(TelechatPreTrainedModel):
    def __init__(self, config: TelechatConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = TelechatModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TELECHAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in TELECHAT and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    The TELECHAT Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    TELECHAT_START_DOCSTRING,
)
class TelechatForQuestionAnswering(TelechatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TelechatModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TELECHAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )