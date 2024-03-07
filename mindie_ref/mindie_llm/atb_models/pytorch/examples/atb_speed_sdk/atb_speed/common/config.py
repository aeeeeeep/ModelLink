import ast
import configparser
import os
from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class PrecisionConfig:
    work_dir: str = ""
    batch: int = 1
    shot: int = 5
    seq_len_out: int = 32
    mode: str = "ceval"

    def __post_init__(self):
        int_attr = ("batch", "shot", "seq_len_out")
        for attr in int_attr:
            self.__dict__[attr] = int(self.__dict__[attr])
        self.work_dir = os.path.realpath(self.work_dir)


@dataclass
class PerformanceConfig:
    model_name: str = ""
    batch_size: int = 1
    max_len_exp: int = 11
    min_len_exp: int = 5
    case_pair: Union[Optional[List[int]], str] = None
    save_file_name: str = ""
    perf_mode: str = "normal"
    skip_decode: int = 0

    def __post_init__(self):
        int_attr = ("batch_size", "max_len_exp", "min_len_exp", "skip_decode")
        for attr in int_attr:
            self.__dict__[attr] = int(self.__dict__[attr])
        if self.case_pair is not None:
            self.case_pair = ast.literal_eval(self.case_pair)


@dataclass
class ModelConfig:
    model_path: str = ""
    device_ids: str = "0"
    parallel_backend: str = "hccl"
    device_num: int = 1
    log_dir: str = os.path.join(os.getcwd(), "atb_speed_log")
    bind_cpu: int = 1

    def __post_init__(self):
        self.model_path = os.path.realpath(self.model_path)
        self.device_num = len(self.device_ids.split(","))
        int_attr = ("bind_cpu",)
        for attr in int_attr:
            self.__dict__[attr] = int(self.__dict__[attr])


@dataclass
class Config:
    model: ModelConfig = None
    performance: PerformanceConfig = None
    precision: PrecisionConfig = None

    def init_config(self, raw_content_path, allow_modify=False):
        if not os.path.exists(raw_content_path):
            raise FileNotFoundError(f"{raw_content_path} not exists.")

        section_map = {
            "model": ModelConfig,
            "performance": PerformanceConfig,
            "precision": PrecisionConfig
        }
        if allow_modify:
            warn_msg = "Warning, allow_modify has been set as True. " \
                       "It is dangerous to modify the reserved fields below.\n"
            for cfg_key, cfg_cls in section_map.items():
                warn_msg = warn_msg + "\n".join(
                    f"{cfg_key}.{sub_k} is reserved."
                    for sub_k in cfg_cls.__dict__ if not sub_k.startswith("__")) + "\n"
            print(warn_msg)
        conf = configparser.ConfigParser()
        conf.read(raw_content_path, encoding="utf-8")
        for section_name, section_content in conf.items():
            if section_name == "DEFAULT":
                continue
            if section_name == "ceval":
                import warnings
                warnings.warn(
                    "The section_name [ceval] is deprecated, please refer to readme and use [precision] instead",
                    DeprecationWarning,
                    stacklevel=2)
                section_name = "precision"
            if not hasattr(self, section_name) and not allow_modify:
                continue
            setattr(self, section_name, section_map.get(section_name)(**section_content))


atb_speed_config = Config()
