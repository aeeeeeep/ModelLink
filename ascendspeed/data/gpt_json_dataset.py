#!/usr/bin/env python3
# coding=utf-8
"""
Author: changwanli
since: 2023-11-16 14:33:00
LastTime: 2023-11-24 10:51:54
LastAuthor: changwanli
message: 
Copyright (c) 2023 Wuhan Artificial Intelligence Research. All Rights Reserved 
"""
import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from glob import glob
from typing import Dict, Sequence
import copy

import numpy as np
import torch
import transformers
from ascendspeed import get_args, is_rank_0, print_rank_0
from ascendspeed.data.dataset_utils import (
    get_split_by_range_,
    get_train_valid_test_split_,
)
from ascendspeed.error_utils import check_divisible, check_equal, ensure_valid
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .utils import LazyFiles

IGNORE_INDEX = -100
PREFIX = ""


class LmDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_seq_length=1024,
        padding=False,
        lazy_mode=False,
        shuffle=False,
        index_ratio_range=None,
        seed=42,
        verbose = True
    ):
        """
        _summary_

        Args:
            data_path (_type_): file path or dir path
            tokenizer (_type_): _description_
            max_seq_length (int, optional): _description_. Defaults to 1024.
            padding (bool, optional): _description_. Defaults to False.
            lazy_mode (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """
        if not os.path.exists(data_path):
            raise ValueError(f"{data_path} is not existed.")

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.padding = padding
        self.lazy_mode = lazy_mode
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.index_ratio_range = index_ratio_range
        self.seed = seed
        self.verbose = verbose

        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = True
        self.preprocess_dict = {
            "pretrain": self.pretrain_data_preprocess,
            "sft": self.sft_data_preprocess,
        }

        self.data = self.load_data(data_path, lazy_mode)
        self.idx_map = self.build_index_map()

    def build_index_map(self):
        idx_map = list(range(len(self.data)))
        if self.shuffle:
            random.Random(self.seed).shuffle(idx_map)
        if self.index_ratio_range is not None:
            index_range = [int(len(self.data) * r) for r in self.index_ratio_range]
            idx_map = idx_map[index_range[0] : index_range[1]]
        return idx_map
    
    def rebuild_index_map(self,index_ratio_range):
        self.index_ratio_range = index_ratio_range
        self.idx_map = self.build_index_map()
    

    def load_data(self, data_path, lazy_mode):
        if os.path.isfile(data_path):
            data_path_list = [data_path]
        else:
            data_path_list = glob(os.path.join(data_path, "*"))
            data_path_list = sorted(data_path_list)
        assert len(data_path_list) > 0, f"data_path {data_path} is empty."
        if lazy_mode:
            return LazyFiles(data_path_list, map_func=json.loads, verbose=self.verbose)
        else:
            data = []
            for data_path in data_path_list:
                print_rank_0(f"Loading {data_path}")
                with open(data_path, "r", encoding="utf8") as fp:
                    for line in fp:
                        info = json.loads(line)
                        data.append(info)
            return data

    def __len__(self):
        return len(self.idx_map)

    def pretrain_data_preprocess(self, data_item):
        max_length = self.max_seq_length + 1
        if not isinstance(data_item["text"], str):
            print_rank_0(
                f"bad type {type(data_item['text'])} of input data {data_item['text']}, please"
                " check text type"
            )
            text = str(data_item["text"])
        else:
            text = data_item["text"]
        add_special_tokens = not (
            self.tokenizer._bos_token.content in text or self.tokenizer._eos_token.content in text
        )
        input_ids = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
        )["input_ids"]
        labels = input_ids[:]
        attention_mask = [1] * len(input_ids)
        if self.padding and len(input_ids) < max_length:
            input_ids += [self.tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels += [IGNORE_INDEX] * (max_length - len(labels))
            attention_mask += [0] * (max_length - len(attention_mask))
        input_ids = input_ids[:max_length-1]
        labels = labels[1:]
        attention_mask = attention_mask[:max_length-1]
        assert len(input_ids) == self.max_seq_length
        assert len(labels) == self.max_seq_length      
        assert len(attention_mask) == self.max_seq_length
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def sft_data_preprocess(self, data_item):
        max_length = self.max_seq_length + 1
        input_ids = []
        labels = []
        for sent_idx, sentence in enumerate(data_item["conversations"]):
            if not isinstance(sentence["value"], str):
                print_rank_0(
                    f"bad type {type(sentence['value'])} of input data {sentence['value']}, please"
                    " check text type"
                )
                sentence["value"] = str(sentence["value"])
            sentence_from = sentence["from"].lower()

            sentence_value = (
                f"###问题：\n{sentence['value']}\n\n###答案:"
                if sentence_from == "question"
                else sentence["value"]
            )
            if sent_idx == 0:
                sentence_value = PREFIX + sentence_value

            sentence_ids = self.tokenizer.encode(
                sentence_value,
                add_special_tokens=False,
            )

            # add bos at every beginning of question
            if sentence_from == "question":
                sentence_ids = [self.tokenizer.bos_token_id] + sentence_ids
            label = (
                copy.deepcopy(sentence_ids)
                if sentence_from != "question"
                else [IGNORE_INDEX] * len(sentence_ids)
            )
            input_ids += sentence_ids
            labels += label
            # add eos at every end of human sentence
            if sentence_from != "question":
                # make sure eos_token_id is correct
                input_ids += [self.tokenizer.eos_token_id]
                labels += [self.tokenizer.eos_token_id]

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = [1] * len(input_ids)

        if self.padding and len(input_ids) < max_length:
            input_ids += [self.tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels += [IGNORE_INDEX] * (max_length - len(labels))
            attention_mask += [0] * (max_length - len(attention_mask))
        input_ids = input_ids[:max_length-1]
        labels = labels[1:]
        attention_mask = attention_mask[:max_length-1]
        # print_rank_0(f' len of input_ids:{len(input_ids)}, labels:{len(labels)}, attention_mask:{len(attention_mask)},')
        assert len(input_ids) == self.max_seq_length, f"{len(input_ids)} != {self.max_seq_length}"
        assert len(labels) == self.max_seq_length, f"{len(labels)} != {self.max_seq_length}"
        assert len(attention_mask) == self.max_seq_length, f"{len(attention_mask)} != {self.max_seq_length}"
    
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

    def get_data_type(self, data_item):
        if "conversations" in data_item:
            return "sft"
        elif "text" in data_item:
            return "pretrain"
        else:
            raise ValueError("unsupported task type.")

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = self.idx_map[index]
        data_item = self.data[index]
        data_type = self.get_data_type(data_item)
        feature_dict = self.preprocess_dict[data_type](data_item)
        feature_dict = {k: np.array(v, dtype=np.int64) for k, v in feature_dict.items()}
        return feature_dict


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        labels = [torch.tensor(instance["labels"]) for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        res = dict(
            input_ids=input_ids,
            labels=labels,
        )

        if "attention_mask" in instances[0]:
            attention_mask = [torch.tensor(instance["attention_mask"]) for instance in instances]
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
            res["attention_mask"] = attention_mask

        return res


def get_train_valid_test_split_ratio(splits_string):
    """Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    ensure_valid(splits_sum > 0.0)
    splits = [split / splits_sum for split in splits]
    splits_ratio = []
    for split in splits:
        if splits_ratio:
            splits_ratio.append(splits_ratio[-1] + split)
        else:
            splits_ratio.append(split)
    return splits_ratio


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets  ...")
    split_ratio = get_train_valid_test_split_ratio(args.split)
    train_ds, valid_ds, test_ds = None, None, None
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=True,
        use_fast=not args.tokenizer_not_use_fast,
    )
    train_ds = LmDataset(
        data_path=args.data_path[0],
        tokenizer=tokenizer,
        max_seq_length=args.seq_length,
        padding=True,
        lazy_mode=True,
        shuffle=True,
        seed=args.seed,
        verbose=is_rank_0()
    )

    valid_ds = []
    test_ds = []
    
    # train_ds = LmDataset(
    #     data_path=args.data_path[0],
    #     tokenizer=tokenizer,
    #     max_seq_length=args.seq_length,
    #     padding=True,
    #     lazy_mode=True,
    #     shuffle=True,
    #     index_ratio_range=[0, split_ratio[0]],
    #     seed=args.seed,
    #     verbose=is_rank_0()
    # )
    # valid_ds = LmDataset(
    #     data_path=args.data_path[0],
    #     tokenizer=tokenizer,
    #     max_seq_length=args.seq_length,
    #     padding=True,
    #     lazy_mode=True,
    #     shuffle=True,
    #     index_ratio_range=[split_ratio[0], split_ratio[1]],
    #     seed=args.seed,
    # )

    # test_ds = LmDataset(
    #     data_path=args.data_path[0],
    #     tokenizer=tokenizer,
    #     max_seq_length=args.seq_length,
    #     padding=True,
    #     lazy_mode=True,
    #     shuffle=True,
    #     index_ratio_range=[split_ratio[1], split_ratio[2]],
    #     seed=args.seed,
    # )
    
    # valid_ds = copy.deepcopy(train_ds)
    # valid_ds.rebuild_index_map([split_ratio[0], split_ratio[1]])
    
    # test_ds = copy.deepcopy(train_ds)
    # test_ds.rebuild_index_map([split_ratio[1], split_ratio[2]])
    
    
    # valid_ds = train_ds
    # valid_ds.rebuild_index_map([split_ratio[0], split_ratio[1]])
    
    # test_ds = train_ds
    # test_ds.rebuild_index_map([split_ratio[1], split_ratio[2]])  
    
    
    
    for idx, feat in enumerate(train_ds):
        if idx == 2:
            break
        else:
            for k,v in feat.items():
                print_rank_0(f"{k}: {v.tolist()}")

    
    print_rank_0(f'total train_ds: {len(train_ds)}')
    print_rank_0(f'total valid_ds: {len(valid_ds)}')
    print_rank_0(f'total test_ds: {len(test_ds)}')

    print_rank_0("> finished creating datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/data/changwanli/data/llm/baichuan13b/2023_09_27_16_02_53-fp16",
        trust_remote_code=True,
        use_fast=True,
    )
    lm_dataset = LmDataset(
        data_path="/mnt/data/changwanli/data/demo_data/sft/sft.jsonl",
        tokenizer=tokenizer,
        max_seq_length=2048,
        padding=True,
        lazy_mode=False,
    )
    # for idx, item in enumerate(lm_dataset):
    #     for k, v in item.items():
    #         print(k, v.shape)
    #     if idx > 10:
    #         break

    print(tokenizer.pad_token_id)
