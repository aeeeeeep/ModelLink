# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import importlib
from dataclasses import dataclass
from typing import Optional, Any

import torch

from transformers import AutoTokenizer, AutoConfig
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from ..llama.modeling_llama import LlamaConfig
from ..deepseek.flash_causal_deepseek import DeepseekConfig
from ..mixtral.flash_causal_mixtral import MixtralConfig
from ..qwen.modeling_qwen import QwenConfig
from ..starcoder.flash_causal_starcoder import StarcoderConfig
from ..telechat.config import TelechatConfig
from ..gpt_neox.config import GPTNeoXConfig
from ..internlm.configuration_internlm import InternLMConfig
from ..aquila.v1_7b.modeling_aquila import AquilaConfig


@dataclass
class BaseRouter:
    model_name_or_path: str = ""
    max_position_embeddings: Optional[int] = None,
    is_flash_causal_lm: bool = True
    revision: Optional[str] = None
    trust_remote_code: bool = None
    use_refactor: bool = False

    # 初始化默认读取的autoconfig，各个模型可能会自定义，self.config会返回后续使用的config，注意不要循环依赖
    config_dict: Any = None
    _config: Any = None
    _tokenizer: Any = None
    _model_cls: Any = None
    is_inited: bool = False

    def __post_init__(self):
        self.model_type = self.config_dict['model_type']
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
                self._config = config_cls.from_dict(self.config_dict)
            except Exception:
                self._config = PretrainedConfig.from_dict(self.config_dict)
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
    def model_version(self):
        """
        次级模型:此处用于区分7b、13b,以small为浮点量化归一版本的tag
        """
        if not self.use_refactor and self.config_dict['num_hidden_layers'] in [32, 40]:
            model_ver = "small"
        else:
            model_ver = ""
        return model_ver

    @property
    def config(self):
        config = LlamaConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        return config
    
    def get_tokenizer(self):
        if self.config_dict['num_hidden_layers'] in [60]:
            # LLaMa 33B use_fast需要使用False
            use_fast = False
        else:
            use_fast = True
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast
        )


@dataclass
class TelechatRouter(BaseRouter):

    @property
    def config(self):
        config = TelechatConfig.from_pretrained(self.model_name_or_path,
                                             revision=self.revision,
                                             trust_remote_code=self.trust_remote_code)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        return config

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )


@dataclass
class DeepseekRouter(BaseRouter):
    @property
    def config(self):
        config = DeepseekConfig.from_pretrained(self.model_name_or_path, torch_dtype=torch.float16)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        return config
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
            use_fast=False
        )
    

@dataclass
class MixtralRouter(BaseRouter):
    @property
    def config(self):
        config = MixtralConfig.from_pretrained(self.model_name_or_path, torch_dtype=torch.float16)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        return config
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
            use_fast=False
        )
    

@dataclass
class StarcoderRouter(BaseRouter):

    @property
    def config(self):
        config = StarcoderConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.seq_length = self.max_position_embeddings
        return config


@dataclass
class BaichuanRouter(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称，比如v2_13b
        :return:
        """
        if self.config_dict['num_hidden_layers'] == 40:  # 只有13b才是40层，同时兼容 v1 v2
            model_ver = "v2_13b"
        else:
            model_ver = "v2_7b"
        return model_ver

    @property
    def config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.model_max_length = self.max_position_embeddings  # 13b
            config.max_position_embeddings = self.max_position_embeddings
        return config

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False
        )


@dataclass
class ChatglmRouter(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称，比如v2_13b
        :return:
        """
        if 'multi_query_attention' in self.config_dict:
            model_ver = "v2_6b"
        else:
            model_ver = "v1_6b"

        return model_ver

    @property
    def config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.seq_length = self.max_position_embeddings
        return config

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False
        )


@dataclass
class QwenRouter(BaseRouter):
    @property
    def config(self):
        return QwenConfig.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code
        )

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=True
        )


@dataclass
class AquilaRouter(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称
        :return:
        """
        return "v1_7b"

    @property
    def config(self):
        config = AquilaConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        return config

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            pad_token='<|endoftext|>',
            padding_side="left",
            trust_remote_code=True,
            use_fast=True
        )


@dataclass
class Gpt_neoxRouter(BaseRouter):
    @property
    def config(self):
        config = GPTNeoXConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.model_max_length = self.max_position_embeddings
        return config

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            truncation_side="left",
            padding_side="left",
            trust_remote_code=True,
        )


@dataclass
class InternlmRouter(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称
        :return:
        """
        return "20b"

    @property
    def config(self):
        config = InternLMConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        return config

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False
        )


@dataclass
class BloomRouter(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称，比如 7b 176b
        :return:
        """
        if self.config_dict['n_layer'] == 30:  # 7b 30层, 176b 70层
            model_ver = "7b"
        else:
            model_ver = "176b"
        return model_ver

    @property
    def config(self):
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code
            )
        if self.max_position_embeddings:
            config.seq_length = self.max_position_embeddings
        return config