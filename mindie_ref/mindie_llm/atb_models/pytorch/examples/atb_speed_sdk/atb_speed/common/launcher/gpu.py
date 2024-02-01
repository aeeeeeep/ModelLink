#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2018. All rights reserved
"""
common launcher
"""
import abc
import os
from typing import Dict

import torch

from atb_speed.common.launcher.base import BaseLauncher


class Launcher(BaseLauncher):
    """
    BaseLauncher
    """

    @staticmethod
    def set_torch_env(device_ids, options: Dict = None):
        """

        :param device_ids:
        :param options:
        :return:
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids

    @abc.abstractmethod
    def init_model(self):
        """
        模型初始化
        :return:
        """
        ...


class ParallelLauncher(Launcher):
    """
    TODO: 待实现
    """

    @abc.abstractmethod
    def init_model(self):
        """
        模型初始化
        :return:
        """
        ...

    @staticmethod
    def set_torch_env(device_ids, options: Dict = None):
        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids

    def setup_model_parallel(self):
        device_id_list = [int(item.strip()) for item in self.device_ids.split(",") if item]
        torch.distributed.init_process_group()
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        torch.manual_seed(1)
        return local_rank, world_size
