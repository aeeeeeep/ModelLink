# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import Optional

import torch

from ..models import get_model
from ..utils import bind_cpus, initialize_torch_distributed, Weights
from ..utils.env import ENV
from ..utils.log import logger, print_log


class ModelRunner:
    model = None,
    soc_info = None,
    head_size = None,
    num_heads = None,
    num_kv_heads = None,
    num_layers = None
    device = None,
    dtype = None,

    def __init__(self, model_name_or_path, rank, world_size,
                 quantize=None, dtype=None, kv_cache_dtype=None,
                 max_position_embeddings=None,
                 is_flash_causal_lm: bool = True,
                 revision: Optional[str] = None,
                 trust_remote_code: bool = True,
                 use_refactor: bool = False,
                 ):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.world_size = world_size
        self.quantize = quantize
        self.dtype = dtype
        self.revision = revision
        if ENV.bind_cpu:
            bind_cpus(world_size, rank, ratio=1.0)
        self.model_cls, self.config, self.tokenizer = \
            get_model(model_name_or_path, quantize, max_position_embeddings, is_flash_causal_lm,
                      revision, trust_remote_code, use_refactor)

        setattr(self.config, "use_refactor", use_refactor)

        self.process_group, self.device = initialize_torch_distributed(rank, world_size)

        print_log(rank, logger.info, f'init tokenizer done: {self.tokenizer}')

    def load_weights(self):
        weights = Weights(
            self.model_name_or_path, self.device, self.dtype,
            process_group=self.process_group,
            quantize=self.quantize,
            revision=self.revision,
            extension=".safetensors"
        )
        if self.quantize == 'smooth_quant':
            weights._set_smooth_quant_params(self.model_name_or_path)
        self.model = self.model_cls(self.config, weights)
        if self.dtype in [torch.float16, torch.bfloat16]:
            self.model.to(self.dtype)
        self.model.to(weights.device)

        self.soc_info = self.model.soc_info
        self.head_size = self.model.head_size
        self.num_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.num_layers = self.model.num_layers

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
