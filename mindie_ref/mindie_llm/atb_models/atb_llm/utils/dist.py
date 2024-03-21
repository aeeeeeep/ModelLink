# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from datetime import timedelta
import os
import json

import torch

from .env import ENV
from .log import logger, print_log


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    @staticmethod
    def allreduce(*args, **kwargs):
        return FakeBarrier()

    @staticmethod
    def allgather(inputs, local_tensor, **kwargs):
        for input_ in inputs:
            input_[0].data = local_tensor[0].data
        return FakeBarrier()

    @staticmethod
    def barrier(*args, **kwargs):
        return FakeBarrier()

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def get_rank_table_file():
    return str(os.getenv("RANKTABLEFILE", ""))


def set_device_from_ranktable(rank, rank_table):
    device_found_flag = False
    print_log(rank, logger.info, f'Rank table given, devices selected from json')
    with open(rank_table, 'r', encoding='utf-8') as device_file:
        data = json.load(device_file)

        for server in data["server_list"]:
            for device in server["device"]:
                if int(device["rank_id"]) == rank:
                    device_id = int(device["device_id"])
                    device = torch.device(f"npu:{device_id}")
                    device_found_flag = True
                    break
            if device_found_flag:
                break
    if not device_found_flag:
        raise ValueError(
            f"ERROR: Rank id is not in the rankTableFile, the input rank is "
            f" {rank}"
        )
    return device


def initialize_distributed(rank, npu_id, world_size):
    if npu_id is None:
        npu_id = rank
    rank_table = str(os.getenv("RANKTABLEFILE", ""))
    print_log(rank, logger.info, f'Rank table file location: {rank_table}')
    if rank_table:
        device = set_device_from_ranktable(rank, rank_table)
    else:
        device = torch.device(f"npu:{npu_id}")
    torch.npu.set_device(device)
    logger.info(f"initialize_distributed has been Set")
    return FakeGroup(rank, world_size), device


def initialize_torch_distributed(rank, world_size):
    if torch.npu.is_available():
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL

        rank_table = str(os.getenv("RANKTABLEFILE", ""))
        print_log(rank, logger.info, f'Rank table file location: {rank_table}')
        if rank_table:
            device = set_device_from_ranktable(rank, rank_table)
            torch.npu.set_device(device)
            return FakeGroup(rank, world_size), device

        device = torch.device(f"npu:{rank}")
        torch.npu.set_device(device)

        backend = "hccl"
        options = ProcessGroupHCCL.Options()
        logger.info(f"ProcessGroupHCCL has been Set")
    else:
        backend = "gloo"
        options = None

    if world_size == 1:
        return FakeGroup(rank, world_size), device
    else:
        if not torch.distributed.is_initialized():
            # Call the init process.
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=ENV.pg_init_timeout),
                pg_options=options,
            )
            logger.info(f"rank {rank} init {torch.distributed.is_initialized()}, init_process_group has been activated")
        else:
            logger.info("torch.distributed is already initialized.")

        return torch.distributed.group.WORLD, device
