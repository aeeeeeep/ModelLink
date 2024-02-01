# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from datetime import timedelta

import torch

from atb_llm.common.log.logging import logger
from atb_llm.utils.env import ENV


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


def initialize_torch_distributed(rank, world_size):
    if torch.npu.is_available():
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL

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
            logger.info(f"rank {rank} init_process_group has been activated")
            logger.info(f"rank {rank} init {torch.distributed.is_initialized()}")
        else:
            logger.info("torch.distributed is already initialized.")

        return torch.distributed.group.WORLD, device
