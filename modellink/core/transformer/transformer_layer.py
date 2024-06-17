from functools import wraps
from dataclasses import dataclass, field
from typing import Dict, Union

from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args


@dataclass
class TransformerLayerSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    post_attn_norm: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    post_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


def transformer_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args_pos_norm = get_args()
        if args_pos_norm.post_norm:
            self.post_attn_norm = build_module(
                kwargs["submodules"].post_attn_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
            self.post_mlp_layernorm = build_module(
                kwargs["submodules"].post_mlp_layernorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
    return wrapper


def transformer_layer_forward(self, hidden_states, attention_mask, context=None,
                              context_mask=None,
                              rotary_pos_emb=None,
                              inference_params=None,
                              packed_seq_params=None):

    # hidden_states: [s, b, h]
    args_pos_norm = get_args()
    # Residual connection.
    residual = hidden_states

    # Optional Input Layer norm
    input_layernorm_output = self.input_layernorm(hidden_states)

    # Self attention.
    attention_output_with_bias = self.self_attention(
        input_layernorm_output,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
    )
    if args_pos_norm.post_norm:
        attention_output = self.post_attn_norm(attention_output_with_bias[0])
        attention_output_with_bias = (attention_output, attention_output_with_bias[1])

    # TODO: could we move `bias_dropout_add_exec_handler` itself
    # inside the module provided in the `bias_dropout_add_spec` module?
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm after self-attention
    pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

    # Cross attention.
    attention_output_with_bias = self.cross_attention(
        pre_cross_attn_layernorm_output,
        attention_mask=context_mask,
        key_value_states=context,
        inference_params=inference_params,
    )

    if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
        context = attention_output_with_bias["context"]

    # TODO: could we move `bias_dropout_add_exec_handler` itself
    # inside the module provided in the `bias_dropout_add_spec` module?
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm post the cross-attention.
    pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

    # MLP.
    mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
    if args_pos_norm.post_norm:
        mlp_output = self.post_mlp_layernorm(mlp_output_with_bias[0])
        mlp_output_with_bias = (mlp_output, mlp_output_with_bias[1])

    # TODO: could we move `bias_dropout_add_exec_handler` itself
    # inside the module provided in the `bias_dropout_add_spec` module?
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            mlp_output_with_bias, residual, self.hidden_dropout
        )

    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    return output, context
