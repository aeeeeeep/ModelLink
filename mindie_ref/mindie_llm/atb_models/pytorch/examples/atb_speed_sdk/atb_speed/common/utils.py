#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2018. All rights reserved
"""
utils
"""
import os
from dataclasses import dataclass

import torch


@dataclass
class TorchParallelInfo:
    __is_initialized: bool = False
    __world_size: int = 1
    __local_rank: int = 0

    def __post_init__(self):
        self.try_to_init()

    def try_to_init(self):
        """
        没有初始化的时候，刷新初始化状态以及world_size local_rank
        :return:
        """
        if not self.__is_initialized:
            is_initialized = torch.distributed.is_initialized()
            if is_initialized:
                self.__local_rank = self.get_rank()
                self.__world_size = self.get_world_size()
            self.__is_initialized = is_initialized
        return self.__is_initialized

    @property
    def is_initialized(self):
        return self.__is_initialized

    @property
    def world_size(self):
        self.try_to_init()
        return self.__world_size

    @property
    def local_rank(self):
        self.try_to_init()
        return self.__local_rank

    @property
    def is_rank_0(self) -> bool:
        return self.local_rank == 0

    @staticmethod
    def get_rank() -> int:
        return 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()

    @staticmethod
    def get_world_size() -> int:
        return 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()


def print_rank_0(*args, **kwargs):
    if torch_parallel_info.is_rank_0:
        print(*args, **kwargs)


def load_atb_speed():
    env_name = "ATB_SPEED_HOME_PATH"
    atb_speed_home_path = os.getenv(env_name)
    if atb_speed_home_path is None:
        raise RuntimeError(f"env {env_name} not exist, source set_env.sh")
    lib_path = os.path.join(atb_speed_home_path, "lib", "libatb_speed_torch.so")
    torch.classes.load_library(lib_path)


torch_parallel_info = TorchParallelInfo()
