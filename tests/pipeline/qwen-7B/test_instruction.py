#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

import sys
import os
import time
import numpy as np
import torch
from common import DistributedTest
from utils import ParamConfig, assert_judge
from megatron.legacy.data.dataset_utils import get_train_valid_test_split_
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
import modellink
from modellink.data.mtf_dataset import get_packed_indexed_dataset, MTFDataset
from modellink.data.decoder_packed_mtf_dataset import DecoderPackedMTFDataset, _build_shuffle_idx


class TestInstruction(DistributedTest):
    world_size = 1

    def init(self, config=ParamConfig):
        sys.argv = ([sys.argv[0]] + config.distributed_param + config.network_size + config.tokenizer_param + 
                    config.auxiliary_param + config.instruction_param + config.output_param)
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=None,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})

        self.args = get_args()

    def _build_index_mappings(
        self,
        name,
        data_prefix,
        start_index,
        nb_documents,
        num_samples: int,
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
        _filename.join('_{}_indexmap'.format(name))
        _filename.join('_{}ns'.format(num_samples))
        _filename.join('_{}s'.format(seed))
        shuffle_idx_filename = _filename.join('_decoder_packed_shuffle_idx.npy')
        if os.path.isfile(shuffle_idx_filename):
            os.remove(shuffle_idx_filename)
        # Build the indexed mapping if not exist.
        if not os.path.isfile(shuffle_idx_filename):

            print(' > WARNING: could not find index map files, building '
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
            print(' > elasped time to build and save shuffle-idx and sample-idx mapping'
                            ' (seconds): {:4f}'.format(time.time() - start_time))


        # Load mappings.
        start_time = time.time()
        print(' > loading shuffle-idx mapping from {}'.format(
            shuffle_idx_filename))
        shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
        print('    loaded indexed file in {:3.3f} seconds'.format(
            time.time() - start_time))

        return shuffle_idx, epoch

    def test_train_valid_test_split(self):
        self.init(config=ParamConfig)
        data_prefix = self.args.data_path[0]
        packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix)
        total_num_of_documents = len(list(packed_indexed_dataset.values())[0])

        assert_judge(52002 == total_num_of_documents)

        splits = get_train_valid_test_split_(self.args.split, total_num_of_documents)
        if self.args.train_samples:
            train_samples = self.args.train_samples
        else:
            train_samples = self.args.train_iters * self.args.global_batch_size
        eval_iters = (self.args.train_iters // self.args.eval_interval + 1) * \
                    self.args.eval_iters
        test_iters = self.args.eval_iters
        train_val_test_num_samples = [train_samples,
                                    eval_iters * self.args.global_batch_size,
                                    test_iters * self.args.global_batch_size]

        def build_shuffle_index(index, name):
            shuffle_index = None
            if splits[index + 1] > splits[index]:
                documents = np.arange(start=splits[index], stop=splits[index + 1],
                                    step=1, dtype=np.int32)
                mtf_dataset = MTFDataset(name=name, data_prefix=data_prefix, documents=documents)
                shuffle_index = self._build_index_mappings(name=name, data_prefix=data_prefix, start_index=documents[0], nb_documents=len(documents), num_samples=train_val_test_num_samples[index], seed=self.args.seed)
            return shuffle_index

        train_shuffle_index, train_epoch = build_shuffle_index(0, 'train')
        valid_shuffle_index, valid_epoch = build_shuffle_index(1, 'valid')
        test_shuffle_index, test_epoch = build_shuffle_index(2, 'test')

        ### 数量验证
        assert_judge(abs(len(train_shuffle_index) - 0.9 * total_num_of_documents * train_epoch) <= train_epoch)
        assert_judge(abs(len(valid_shuffle_index) - 0.05 * total_num_of_documents * valid_epoch) <= valid_epoch)
        assert_judge(abs(len(test_shuffle_index) - 0.05 * total_num_of_documents * test_epoch) <= test_epoch)

        ### document划分验证
        train_shuffle_index_set = set(train_shuffle_index)
        valid_shuffle_index_set = set(valid_shuffle_index)
        test_shuffle_index_set = set(test_shuffle_index)

        assert_judge(len(train_shuffle_index_set & valid_shuffle_index_set) == 0)
        assert_judge(len(test_shuffle_index_set & valid_shuffle_index_set) == 0)
        assert_judge(len(train_shuffle_index_set & test_shuffle_index_set) == 0)


    def test_instruction(self):
        pass