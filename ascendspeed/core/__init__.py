import megatron
from .fused_kernels.rms_norm import rms_norm_forward
from .fused_kernels.transformer import parallel_mlp_init, parallel_mlp_forward

megatron.model.rms_norm.RMSNorm.forward = rms_norm_forward
megatron.model.transformer.ParallelMLP.__init__ = parallel_mlp_init
megatron.model.transformer.ParallelMLP.forward = parallel_mlp_forward
