# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import importlib
from dataclasses import dataclass
from typing import Optional, Any

from transformers import AutoTokenizer

from ..llama.modeling_llama import LlamaConfig
from ..starcoder.flash_causal_starcoder import StarcoderConfig


@dataclass
class BaseRouter:
    model_name_or_path: str = ""
    quantize: Optional[str] = None
    is_flash_causal_lm: bool = True
    revision: Optional[str] = None
    trust_remote_code: bool = None

    # 初始化默认读取的autoconfig，各个模型可能会自定义，self.config会返回后续使用的config，注意不要循环依赖
    ori_config: Any = None
    _config: Any = None
    _tokenizer: Any = None
    _model_cls: Any = None
    is_inited: bool = False

    def __post_init__(self):
        self.model_type = self.ori_config.model_type
        self.model_type_cap = self.model_type.capitalize()

    @property
    def model_version(self):
        """
        次级模型名称，比如v2_13b
        :return:
        """
        return ""

    @property
    def model_cls(self):
        if self._model_cls is None:
            self._model_cls = self.get_model_cls()
        return self._model_cls

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    @property
    def config(self):
        if self._config is None:
            try:
                config_cls = self.get_config_cls()
                self._config = config_cls.from_pretrained(self.model_name_or_path,
                                                          revision=self.revision,
                                                          trust_remote_code=self.trust_remote_code)
            except Exception:
                self._config = self.ori_config
        return self._config

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=True
        )

    def get_config_cls(self):
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        if self.model_version:
            model_file_dir_name = model_file_dir_name + f"{self.model_version}."
        config_file_name = 'config'
        module_path = f"{model_file_dir_name}{config_file_name}"
        module = importlib.import_module(module_path)
        config_cls_name = f"{self.model_type_cap}Config"
        return getattr(module, config_cls_name)

    def get_model_cls(self):
        """
        get_model_cls
        """
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        if self.model_version:
            model_file_dir_name = model_file_dir_name + f"{self.model_version}."
        model_file_name = 'flash_causal' if self.is_flash_causal_lm else 'causal'
        module_path = f"{model_file_dir_name}{model_file_name}_{self.model_type}"
        module = importlib.import_module(module_path)
        model_cls_name = f"{self.model_type_cap}ForCausalLM"
        if self.is_flash_causal_lm:
            model_cls_name = "Flash" + model_cls_name
        return getattr(module, model_cls_name)


@dataclass
class LlamaRouter(BaseRouter):

    @property
    def config(self):
        return LlamaConfig.from_pretrained(self.model_name_or_path,
                                           revision=self.revision,
                                           trust_remote_code=self.trust_remote_code)


@dataclass
class StarcoderRouter(BaseRouter):

    @property
    def config(self):
        return StarcoderConfig.from_pretrained(self.model_name_or_path,
                                           revision=self.revision,
                                           trust_remote_code=self.trust_remote_code)


@dataclass
class BaichuanRouter(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称，比如v2_13b
        :return:
        """
        if self.ori_config.num_hidden_layers == 40:  # 只有13b才是40层，同时兼容 v1 v2
            model_ver = "v2_13b"
        else:
            model_ver = "v2_7b"
        return model_ver

    @property
    def config(self):
        config_cls = self.get_config_cls()
        return config_cls.from_pretrained(self.model_name_or_path,
                                          revision=self.revision,
                                          trust_remote_code=self.trust_remote_code)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False
        )
