# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from .router import router


def get_model(model_name_or_path: str,
              max_position_embeddings: Optional[int] = None,
              is_flash_causal_lm: bool = True,
              revision: Optional[str] = None,
              trust_remote_code: bool = True,
              use_refactor: bool = False,
              ):
    config_dict, kwargs = PretrainedConfig.get_config_dict(model_name_or_path)
    router_cls = getattr(router, f"{config_dict['model_type'].capitalize()}Router")
    router_ins = router_cls(
        model_name_or_path,
        max_position_embeddings,
        is_flash_causal_lm,
        revision,
        trust_remote_code,
        use_refactor,
        config_dict)
    config = router_ins.config
    if not hasattr(config, 'quantize'):
        setattr(config, 'quantize', None)
    return router_ins.model_cls, config, router_ins.tokenizer
