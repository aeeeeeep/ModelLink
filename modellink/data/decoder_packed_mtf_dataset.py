# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import time
import logging

import numpy as np
import torch

from megatron import print_rank_0, get_args
from modellink.utils import is_rank_0
from modellink.tokenizer import build_tokenizer
from megatron.core import parallel_state
from megatron.data.dataset_utils import get_train_valid_test_split_
from modellink.data.mtf_dataset import MTFDataset, get_packed_indexed_dataset
from modellink.error_utils import check_equal

logger = logging.getLogger(__name__)


def build_train_valid_test_datasets(
    data_prefix,
    splits_string,
    seq_length: int,
    train_valid_test_num_samples,
    seed,
):
    """Build train, valid, and test datasets."""

    args = get_args()

    tokenizer = build_tokenizer(args)
    pad_token = tokenizer.pad
    eos_token = tokenizer.eos
    
    # Only Support Single dataset.
    all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(
        data_prefix=data_prefix[0],
        splits_string=splits_string,
        seq_length=seq_length,
        pad_token=pad_token,
        eos_token=eos_token,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seed=seed,
    )

    return all_train_datasets, all_valid_datasets, all_test_datasets


def _build_train_valid_test_datasets(
    data_prefix,
    splits_string,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
):
    """Build train, valid, and test datasets."""

    # Target indexed dataset.
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix)

    total_num_of_documents = len(list(packed_indexed_dataset.values())[0])
    # splits here is an array of size 4  [train_start_index, valid_start_index, test_start_index, test_end_index]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = DecoderPackedMTFDataset(
                name=name,
                data_prefix=data_prefix,
                documents=documents,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                num_samples=train_valid_test_num_samples[index],
                seed=seed
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


class DecoderPackedMTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        documents,
        num_samples,
        seq_length: int,
        pad_token: int,
        eos_token: int,
        seed,
    ):
        args = get_args()
        self.mtf_dataset = MTFDataset(name=name, data_prefix=data_prefix, documents=documents)

        self.pad_token = pad_token
        self.seq_length = seq_length

        self.shuffle_index = _build_index_mappings(name=name, data_prefix=data_prefix, start_index=documents[0], nb_documents=len(documents), mtf_dataset=self.mtf_dataset, num_samples=num_samples, seq_length=seq_length, seed=seed)

    def __len__(self):
        return len(self.shuffle_index)

    def __getitem__(self, idx):
        doc_idx = self.shuffle_index[idx]
        item = self.mtf_dataset[doc_idx]
        return {
            "input_ids": self._cut_token(item["input_ids"][:-1], np.int64),
            "attention_mask": self._cut_token(item["attention_mask"][:-1], np.int64),
            "labels": self._cut_token(item["labels"][1:], np.int64),
        }
    
    def _cut_token(self, token, dtype):
        token_length = len(token)
        if token_length >= self.seq_length:
            token = token[:self.seq_length]
        return token.astype(dtype)


def _build_index_mappings(
    name,
    data_prefix,
    start_index,
    nb_documents,
    mtf_dataset,
    num_samples: int,
    seq_length: int,
    seed,
):
    """
    - `shuffle_index` is [num_epoch * len(self.mtf)]
    - `sample_index` is [num_sample, 2] (storing the start and end of the sample). We query the sample via `self.shuffle_index[start:end]`
    """

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}s'.format(seed)
    shuffle_idx_filename = _filename + '_decoder_packed_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() % torch.cuda.device_count() == 0:
        if not os.path.isfile(shuffle_idx_filename):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # iteratively add the entire dataset for every epoch and see if it's enough given current packing strategy
            start_time = time.time()
            epoch = 0
            shuffle_idx = []
            while len(shuffle_idx) <= num_samples:
                new_document_ids = _build_shuffle_idx(nb_documents=nb_documents, start_index=start_index, np_rng=np_rng)
                # Generate a shuffling of the entire dataset
                shuffle_idx.extend(new_document_ids.tolist())
                epoch += 1

            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx and sample-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
    item = (torch.distributed.get_world_size() //
            torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group()))
    check_equal(counts[0].item(), item)

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))

    return shuffle_idx


def _build_shuffle_idx(nb_documents: int, start_index, np_rng):
    """Build the range [0, dataset_size) and shuffle."""
    dtype_ = np.int64
    stop = start_index + nb_documents

    result = np.arange(start=start_index, stop=stop, step=1, dtype=dtype_)

    # in-place shuffling
    np_rng.shuffle(result)

    return result
