# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import logging
from typing import Any, Optional, Type, Union

import torch

from megatron.training import get_args
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset

from ..parallel_state import get_pipeline_model_parallel_node_info

logger = logging.getLogger(__name__)

DistributedDataset = Union[BlendedDataset, MegatronDataset, IndexedDataset]


def need_to_build_dataset():
    args = get_args()
    share_save = not args.no_shared_storage
    rank = torch.distributed.get_rank()
    if share_save:
        return rank == 0    
    gpus_per_node = torch.cuda.device_count()
    node_pp_group_info = get_pipeline_model_parallel_node_info()
    flag = False
    num_edge_ranks = sum([x != 1 for x in node_pp_group_info])
    if num_edge_ranks >= 1:
        first_idx = node_pp_group_info.index([x for x in node_pp_group_info if x != 1][0])
        flag = (first_idx == rank % gpus_per_node)
    return flag


def _build_generic_dataset(
    self, cls: Type[DistributedDataset], *args: Any,
) -> Optional[DistributedDataset]:
    """Build the DistributedDataset

    Return None if and only if the underlying MegatronDataset class is not built on the current
    rank and torch.distributed is initialized.

    Args:
        cls (Type[DistributedDataset]): The DistributedDataset class to be built

        args (Tuple[Any]): The positional arguments used to build the provided
        DistributedDataset class

    Raises:
        Exception: When the dataset constructor raises an OSError

    Returns:
        Optional[DistributedDataset]: The DistributedDataset instantion or None
    """
    if torch.distributed.is_initialized():

        dataset = None

        # First, build on ranks of first and last of pp group
        to_build_flag = need_to_build_dataset()
        if to_build_flag and getattr(self.config, "is_built_on_rank")():
            try:
                dataset = cls(*args)
            except OSError as err:
                log = (
                    f"Failed to write dataset materials to the data cache directory. "
                    + f"Please supply a directory to which you have write access via "
                    + f"the path_to_cache attribute in BlendedMegatronDatasetConfig and "
                    + f"retry. Refer to the preserved traceback above for more information."
                )
                raise Exception(log) from err

        torch.distributed.barrier()

        # After, build on other ranks
        if not to_build_flag and getattr(self.config, "is_built_on_rank")():
            dataset = cls(*args)

        return dataset

    return cls(*args)

