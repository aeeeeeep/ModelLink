
import torch
import torch.nn.functional as F
import megatron
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.transformer import ParallelMLP


def parallel_mlp_init(self, config, is_expert=False):
    super(ParallelMLP, self).__init__()
    args = get_args()
    
    self.add_bias = config.add_bias_linear
    ffn_hidden_size = config.ffn_hidden_size
    self.gated_linear_unit = config.gated_linear_unit

    self.layer_fusion = args.mlp_layer_fusion
    if config.gated_linear_unit or self.layer_fusion:
        ffn_hidden_size *= 2
    if (sum([args.add_gate, config.gated_linear_unit]) > 1):
        raise Exception(f"only can use one method in [add_gate :"
        f"{args.add_gate},gated_linear_unit :{config.gated_linear_unit}],")
    
    self.add_gate = args.add_gate

    if self.add_gate:
        if self.layer_fusion:
            self.proj = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                gather_output=False,
                skip_bias_add=True,
                is_expert=is_expert,
            )
        else:
            self.gate_proj = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                gather_output=False,
                skip_bias_add=True,
                is_expert=is_expert,
            )

    if not self.layer_fusion:
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
            is_expert=is_expert,
        )

    self.bias_gelu_fusion = False
    self.activation_func = None
    self.swiglu = args.swiglu

    if args.openai_gelu:
        self.activation_func = openai_gelu
    elif args.onnx_safe:
        self.activation_func = erf_gelu
    elif args.swiglu:
        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]
        self.activation_func = swiglu
    elif args.squared_relu:
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        self.activation_func = squared_relu
    else:
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu

    # Project back to h.
    self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
        config.ffn_hidden_size,
        config.hidden_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=self.add_bias,
        skip_bias_add=True,
        input_is_parallel=True,
        is_expert=is_expert,
    )

def parallel_mlp_forward(self, hidden_states):

    if self.add_gate:
        if self.layer_fusion:
            gate_and_up_proj = self.proj(hidden_states)[0]
            (gate, up_proj) = tensor_parallel.utils.split_tensor_along_last_dim(
                gate_and_up_proj, 2, contiguous_split_chunks=True)
            intermediate_parallel = F.silu(gate) * up_proj
        else:
            intermediate_parallel = F.silu(
                self.gate_proj(hidden_states)[0]) * self.dense_h_to_4h(hidden_states)[0]

    else:
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        if self.bias_gelu_fusion:
            assert self.add_bias is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

    # [s, b, h]
    output, output_bias = self.dense_4h_to_h(intermediate_parallel)
    return output, output_bias
