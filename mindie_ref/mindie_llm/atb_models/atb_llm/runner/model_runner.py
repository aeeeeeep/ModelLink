# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import torch
from ..models import get_model
from ..utils.env import ENV
from ..utils import bind_cpus, initialize_distributed, Weights
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
                 npu_id=None,
                 local_rank=None,
                 kv_cache_dtype=None,
                 max_position_embeddings=None,
                 is_flash_causal_lm: bool = True,
                 use_refactor: bool = True,
                 ):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.local_rank = local_rank if local_rank is not None else rank
        self.npu_id = npu_id if npu_id is not None else self.local_rank
        self.world_size = world_size

        if ENV.bind_cpu:
            try:
                bind_cpus(world_size, self.npu_id, ratio=1.0)
            except Exception as err:
                logger.error(f"Binding CPU failed\n{err}\n skip.")
        self.model_cls, self.config, self.tokenizer = \
            get_model(model_name_or_path, max_position_embeddings, is_flash_causal_lm,
                      revision=None,
                      trust_remote_code=True,
                      use_refactor=use_refactor)

        setattr(self.config, "use_refactor", use_refactor)
        self.quantize = self.config.quantize
        self.dtype = self.config.torch_dtype

        print_log(rank, logger.info, f'model_runner.quantize: {self.quantize}\n, '
                                     f'model_runner.dytpe: {self.dtype}')

        if self.dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(f'unsupported type: {self.dtype}')

        self.process_group, self.device = initialize_distributed(self.rank, self.npu_id, world_size)

        print_log(rank, logger.info, f'init tokenizer done: {self.tokenizer}')

    def load_weights(self):
        weights = Weights(
            self.model_name_or_path, self.device, self.dtype,
            process_group=self.process_group,
            quantize=self.quantize,
            revision=None,
            extension=".safetensors"
        )
        self.model = self.model_cls(self.config, weights)

        self.model.to(weights.device)

        self.soc_info = self.model.soc_info
        self.head_size = self.model.head_size
        self.num_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.num_layers = self.model.num_layers

        print_log(self.rank, logger.info, f'model:\n {self.model}')

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def save_pretrained(self, **kwargs):
        if 'save_directory' not in kwargs:
            raise ValueError('save_directory is required')
        kwargs['save_directory'] = os.path.join(kwargs['save_directory'], f'part{self.rank}')
        return self.model.save_pretrained(**kwargs)
