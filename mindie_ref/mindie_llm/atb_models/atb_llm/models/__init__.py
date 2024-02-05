# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from .router import router


def get_model(model_name_or_path: str,
              quantize: Optional[str] = None,
              max_position_embeddings: Optional[int] = None,
              is_flash_causal_lm: bool = True,
              revision: Optional[str] = None,
              trust_remote_code: bool = True,
              ):
    config = PretrainedConfig.from_pretrained(model_name_or_path,
                                              revision=revision,
                                              trust_remote_code=trust_remote_code)
    router_cls = getattr(router, f"{config.model_type.capitalize()}Router")
    router_ins = router_cls(
        model_name_or_path,
        quantize,
        max_position_embeddings,
        is_flash_causal_lm,
        revision,
        trust_remote_code,
        config)
    config = router_ins.config

    config.quantize = quantize
    return router_ins.model_cls, config, router_ins.tokenizer
