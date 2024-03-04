# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from dataclasses import dataclass

from .log import logger


@dataclass
class EnvVar:
    """
    环境变量
    """
    # 使用昇腾加速库
    use_ascend: bool = os.getenv("USE_ASCEND", "1") == "1"
    # 最大内存 GB
    max_memory_gb: str = os.getenv("MAX_MEMORY_GB", None)
    reserved_memory_gb: int = int(os.getenv("RESERVED_MEMORY_GB", "3"))
    # 跳过warmup
    skip_warmup: bool = os.getenv("SKIP_WARMUP", "0") == "1"
    # 使用哪些卡
    visible_devices: str = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
    # 使用host后处理
    use_host_chooser: bool = os.getenv("USE_HOST_CHOOSER", "1") == "1"
    # 是否绑核
    bind_cpu: bool = os.getenv("BIND_CPU", "1") == "1"
    # process group 初始化timeout，单位 秒，默认是10
    pg_init_timeout = int(os.getenv("PG_INIT_TIMEOUT", "10"))

    memory_fraction = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))

    profiling_enable = os.getenv("ATB_PROFILING_ENABLE", "0") == "1"
    profiling_filepath = os.getenv("PROFILING_FILEPATH", './profiling')

    benchmark_enable = os.getenv("ATB_LLM_BENCHMARK_ENABLE", "0") == "1"
    benchmark_filepath = os.getenv("ATB_LLM_BENCHMARK_FILEPATH", None)

    logits_save_enable = os.getenv("ATB_LLM_LOGITS_SAVE_ENABLE", "0") == "1"
    logits_save_folder = os.getenv("ATB_LLM_LOGITS_SAVE_FOLDER", './')

    def __post_init__(self):
        logger.info(self.dict())

    def dict(self):
        return self.__dict__


ENV = EnvVar()
