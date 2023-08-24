from ascendspeed.core import parallel_state
from ascendspeed.core import tensor_parallel

from .inference_params import InferenceParams
from .model_parallel_config import ModelParallelConfig

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = ["parallel_state", "tensor_parallel", "utils", "InferenceParams", "ModelParallelConfig"]
