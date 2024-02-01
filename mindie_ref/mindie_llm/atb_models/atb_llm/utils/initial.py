# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from dataclasses import dataclass

import torch
import torch_npu


@dataclass
class NPUSocInfo:
    soc_name: str = ""
    soc_version: int = -1
    need_nz: bool = False

    def __post_init__(self):
        self.soc_version = torch_npu._C._npu_get_soc_version()
        if self.soc_version in (100, 101, 102, 200, 201, 202, 203):
            self.need_nz = True


def load_atb_speed():
    atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
    if atb_speed_home_path is None:
        raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
    lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
    torch.classes.load_library(lib_path)
