import logging
import os
import time
from typing import Tuple

import numpy
import torch

from megatron.core.datasets.utils import log_single_rank
from megatron.core.datasets.gpt_dataset import (_get_num_epochs, _get_num_tokens_per_epoch,
                                                _build_document_index, _build_shuffle_index)
from modellink.error_utils import check_condition

logger = logging.getLogger(__name__)


def _cache_hit_check(description, document_index, sample_index, shuffle_index):
    cache_hit = all(
        map(
            os.path.isfile,
            [
                description,
                document_index,
                sample_index,
                shuffle_index,
            ],
        )
    )
    return cache_hit


def _build_document_sample_shuffle_indices(
        self,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Build the document index, the sample index, and the shuffle index

    The document index:
        -- 1-D
        -- An ordered array of document ids

    The sample index:
        -- 2-D
        -- The document indices and offsets which mark the start of every sample

    The shuffle index:
        -- 1-D
        -- A random permutation of index range of the sample index

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the
        shuffle index

    TODO: Explain the 80% threshold
    """
    path_to_cache = getattr(self.config, "path_to_cache")
    if path_to_cache is None:
        path_to_cache = os.path.join(
            self.indexed_dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
        )

    get_path_to = lambda suffix: os.path.join(
        path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}"
    )
    path_to_description = get_path_to("description.txt")
    path_to_document_index = get_path_to("document_index.npy")
    path_to_sample_index = get_path_to("sample_index.npy")
    path_to_shuffle_index = get_path_to("shuffle_index.npy")
    cache_hit = _cache_hit_check(path_to_description,
                                 path_to_document_index, path_to_sample_index, path_to_shuffle_index)

    num_tokens_per_epoch = _get_num_tokens_per_epoch(self.indexed_dataset, self.indexed_indices)

    sequence_length = getattr(self.config, "sequence_length")

    num_epochs = _get_num_epochs(num_tokens_per_epoch, sequence_length, self.num_samples)
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise ZeroDivisionError
    local_rank = torch.distributed.get_rank() % device_count
    world_siz = torch.distributed.get_world_size()
    if not cache_hit and (local_rank == 0 or (local_rank == device_count - 1 and world_siz > device_count)):
        log_single_rank(
            logger,
            logging.INFO,
            f"Build and save the {type(self).__name__} {self.index_split.name} indices",
        )

        if num_epochs == 1:
            separate_final_epoch = False
        else:
            # Get the number of samples for the last epoch
            num_samples_sans_final_epoch = (
                                                   (num_epochs - 1) * num_tokens_per_epoch - 1
                                           ) // sequence_length
            num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
            num_samples_per_epoch = (num_tokens_per_epoch - 1) // sequence_length

            # num_samples_from_final_epoch should be non-negative
            check_condition(num_samples_from_final_epoch >= 0)

            # num_samples_from_final_epoch should not exceed max value
            check_condition(num_samples_from_final_epoch <= num_samples_per_epoch + 1)

            # Separate the final epoch if it falls below the threshold
            threshold = 0.80
            separate_final_epoch = num_samples_from_final_epoch < int(
                threshold * num_samples_per_epoch
            )

            log_single_rank(
                logger,
                logging.DEBUG,
                f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
            )
            log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
            log_single_rank(
                logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
            )

        log_single_rank(
            logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
        )

        numpy_random_state = numpy.random.RandomState(getattr(self.config, "random_seed"))

        os.makedirs(path_to_cache, exist_ok=True)

        # Write the description
        with open(path_to_description, "wt") as writer:
            writer.write(self.unique_description)

        # Build the document index
        log_single_rank(
            logger,
            logging.INFO,
            f"\tBuild and save the document index to {os.path.basename(path_to_document_index)}",
        )
        t_beg = time.time()
        document_index = _build_document_index(
            self.indexed_indices, num_epochs, numpy_random_state, separate_final_epoch
        )
        numpy.save(path_to_document_index, document_index, allow_pickle=True)
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        # Build the sample index
        log_single_rank(
            logger,
            logging.INFO,
            f"\tBuild and save the sample index to {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        from megatron.core.datasets import helpers

        check_condition(document_index.dtype == numpy.int32)
        check_condition(self.indexed_dataset.sequence_lengths.dtype == numpy.int32)

        sample_index = helpers.build_sample_idx(
            self.indexed_dataset.sequence_lengths,
            document_index,
            sequence_length,
            num_epochs,
            num_tokens_per_epoch,
        )
        numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        # Build the shuffle index
        log_single_rank(
            logger,
            logging.INFO,
            f"\tBuild and save the shuffle index to {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        if separate_final_epoch:
            shuffle_index = _build_shuffle_index(
                num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
            )
        else:
            shuffle_index = _build_shuffle_index(
                sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
            )
        numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    while not cache_hit:
        cache_hit = _cache_hit_check(path_to_description,
                                     path_to_document_index, path_to_sample_index, path_to_shuffle_index)
        time.sleep(1)
    log_single_rank(
        logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
    )

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
    )
    t_beg = time.time()
    document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
    )
    t_beg = time.time()
    sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')
    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
    )
    t_beg = time.time()
    shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
    )
    log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

    return document_index, sample_index, shuffle_index
