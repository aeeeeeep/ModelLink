#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2018. All rights reserved
"""
common launcher
"""
import abc
from dataclasses import dataclass
from typing import Dict

import torch
import torch_npu
from atb_speed.common.config import atb_speed_config
from atb_speed.common.cpu_binding import CPUBinder
from atb_speed.common.launcher.base import BaseLauncher


@dataclass
class NPUSocInfo:
    soc_name: str = ""
    soc_version: int = -1
    need_nz: bool = False

    def __post_init__(self):
        soc_version_map = {-1: "unknown soc version",
                           100: "910PremiumA", 101: "910ProA", 102: "910A", 103: "910ProB", 104: "910B",
                           200: "310P1", 201: "310P2", 202: "310P3", 203: "310P4",
                           220: "910B1", 221: "910B2", 222: "910B2C", 223: "910B3", 224: "910B4",
                           240: "310B1", 241: "310B2", 242: "310B3",
                           250: "910C1", 251: "910C2", 252: "910C3", 253: "910C4"
                           }
        self.soc_version = torch_npu._C._npu_get_soc_version()
        self.soc_name = soc_version_map.get(self.soc_version, soc_version_map[-1])
        if self.soc_version in (100, 101, 102, 103, 104, 200, 201, 202, 203):
            self.need_nz = True


class Launcher(BaseLauncher):
    """
    BaseLauncher
    """

    def __init__(self, device_ids: str = None, model_path="", options=None):
        super().__init__(device_ids, model_path, options)
        self.soc_info = NPUSocInfo()
        self.fit_npu(self.model)

    @staticmethod
    def set_torch_env(device_ids, options: Dict = None):
        """

        :param device_ids:
        :param options:
        :return:
        """
        torch_npu.npu.set_device(int(device_ids.split(",")[0]))
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_option(options)

    def fit_npu(self, model):
        """
        芯片适配,提前转换，提高性能
        :param model:
        :return:
        """
        if not self.soc_info.need_nz:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
            self.logger.info(f"soc info: {self.soc_info.soc_version} , {self.soc_info.soc_name}, support ND")
        else:
            # if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if name == 'lm_head':
                        # eliminate TransData op before lm_head calculation
                        module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                    module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
            self.logger.info(f"soc info: {self.soc_info.soc_version} , {self.soc_info.soc_name}, support NZ")

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)

    @abc.abstractmethod
    def init_model(self):
        """
        模型初始化
        :return:
        """
        ...

    def bind_cpu(self):
        """
        绑核
        :return:
        """
        cpu_binder = CPUBinder(self.logger)
        cpu_binder.bind_cpus(self.device_id_list, self.local_rank, 1.0)
        self.logger.info("Bind cpu successfully!")


class ParallelLauncher(Launcher):

    @abc.abstractmethod
    def init_model(self):
        """
        模型初始化
        :return:
        """
        ...

    @staticmethod
    def set_torch_env(device_ids, options: Dict = None):
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_option(options)

    def setup_model_parallel(self):
        torch.distributed.init_process_group(atb_speed_config.model.parallel_backend)
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        torch_npu.npu.set_device(self.device_id_list[local_rank])
        # seed must be the same in all processes
        torch.manual_seed(1)
        return local_rank, world_size
