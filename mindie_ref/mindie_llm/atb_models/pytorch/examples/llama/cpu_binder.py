# Copyright 2023-2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass
from typing import List, Dict, Union
import psutil


@dataclass
class DeviceInfo:
    _info_line: str = ""
    npu_id: int = 0
    chip_id: int = 0
    chip_logic_id: Union[int, str] = 0
    chip_name: str = ""

    def __post_init__(self):
        self.npu_id, self.chip_id, self.chip_logic_id, self.chip_name = self._info_line.strip().split(None, 3)
        self.npu_id = int(self.npu_id)
        self.chip_id = int(self.chip_id)
        if self.chip_logic_id.isnumeric():
            self.chip_logic_id = int(self.chip_logic_id)


def get_device_map_info():
    device_map_info = {}
    device_map = os.popen(f"npu-smi info -t board -m").read().strip().split("\n")[1:]
    for line in device_map:
        device_info = DeviceInfo(line.strip())
        if isinstance(device_info.chip_logic_id, int):
            device_map_info[device_info.chip_logic_id] = device_info
    return device_map_info


def get_pcie_info(devices, keyword="PCIeBusInfo"):
    device_map_info = get_device_map_info()
    device_pcie_tbl = {}
    for device in devices:
        device_info = device_map_info.get(device)
        if not device_info:
            raise RuntimeError("Can not get device info, you can use BIND_CPU=0 to skip.")
        pcie_info = os.popen(
            f"npu-smi info -t board -i {device_info.npu_id} -c {device_info.chip_id}"
        ).read().strip().split("\n")
        for _ in pcie_info:
            line = ''.join(_.split()) # 此处是因为310P的关键字是 PCIe Bus Info 910是 PCIeBusInfo，故去掉空格以此兼容
            if line.startswith(keyword):
                device_pcie_tbl[device] = line[len(keyword) + 1:]
                break

    return device_pcie_tbl


def get_numa_info(pcie_tbl, keyword="NUMAnode"):
    device_numa_tbl = {}  # key is device id, value is numa id
    numa_devices_tbl = {}  # key is numa id, value is device id list

    for device, pcie_no in pcie_tbl.items():
        numa_info = os.popen(f"lspci -s {pcie_no} -vvv").read().strip().split("\n")
        for _ in numa_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                numa_id = int(line[len(keyword) + 1:])
                device_numa_tbl[device] = numa_id

                devices = numa_devices_tbl.get(numa_id, None)
                if devices is None:
                    numa_devices_tbl[numa_id] = list()

                numa_devices_tbl[numa_id].append(device)
                break

    return device_numa_tbl, numa_devices_tbl


def get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
    cpu_idx_tbl = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    cpu_info = os.popen(f"lscpu").read().strip().split("\n")
    for _ in cpu_info:
        line = ''.join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    raise Exception("lscpu command output error, please check !")

                ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]

            numa_id = int(split_info[0].replace(keyword1, '').replace(keyword2, ''))
            cpu_idx_tbl[numa_id] = ranges
    return cpu_idx_tbl


def bind_cpus(world_size, rank_id, device_id, ratio=0.5):
    """
    可以用export CPU_BINDING_NUM设置每个进程绑的核数;如果不设置CPU_BINDING_NUM,
    会根据ratio(numa利用率)进行计算,如果有64个核，0.5表示用一半，用32个核, 平分给亲和在这个numa上的npu
    """
    devices = [_ for _ in range(device_id, device_id + world_size)]

    # 获取npu和pcie的对应关系
    device_pcie_tbl = get_pcie_info(devices)
    # 根据pcie信息获取npu和numa的对应关系
    device_numa_tbl, numa_devices_tbl = get_numa_info(device_pcie_tbl)
    if numa_devices_tbl is None:
        return
    # 获取使用的numa对应的cpu核分配信息
    cpu_idx_tbl = get_cpu_info(list(numa_devices_tbl.keys()))

    # 当前rank的npu id
    cur_device = rank_id + device_id
    # 获取npu对应的numa id
    numa_id = device_numa_tbl[cur_device]

    # 获取共享该numa的npu信息
    shard_devices = numa_devices_tbl[numa_id]
    # 按照npu id进行排序
    shard_devices.sort()

    # 获取该numa上所有的cpu id信息
    all_cpus = cpu_idx_tbl[numa_id]

    cpu_nums = len(all_cpus)
    # 计算给该共享numa的npu分配的核的个数
    cpu_binding_num = os.environ.get("CPU_BINDING_NUM", None)
    if cpu_binding_num is None:
        cpu_num_per_device = int(cpu_nums * ratio // len(shard_devices))
    else:
        cpu_num_per_device = int(cpu_binding_num)
        if len(shard_devices) * cpu_num_per_device > cpu_nums:
            raise Exception(
                f"Cpu num in numa {numa_id} to assign {cpu_num_per_device} for every device is not enough, "
                f"please decrease the value of CPU_BINDING_NUM!")

    # 获取该npu的下标信息
    idx = shard_devices.index(cur_device)
    # 给该npu分配要绑定的cpu id
    binding_cpus = [all_cpus[_] for _ in range(idx * cpu_num_per_device, (idx + 1) * cpu_num_per_device)]

    # cpu bind
    p = psutil.Process()
    p.cpu_affinity(binding_cpus)
    new_affinity = p.cpu_affinity()
    print("binding cpu success!")
