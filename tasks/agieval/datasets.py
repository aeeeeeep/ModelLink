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

"""Agi-Eval datasets."""
import os
import json
import math
import random

import torch
import numpy as np

from ascendspeed import get_args
from ascendspeed import print_rank_0
from ascendspeed import get_tokenizer

def build_dataset(task_type, task):
    """Helper function to select and build dataset."""
    dataset = None
    if task_type == 'chinese_qa':
        dataset =  _ChineseQADataset(_get_data_path(task))
    if task_type == 'english_qa':
        dataset =  _EnglishQADataset(_get_data_path(task))
    if task_type == 'english_cloze':
        dataset = _EnglishClozeDataset(_get_data_path(task))
    if task_type == 'chinese_cloze':
        dataset = _ChineseClozeDataset(_get_data_path(task))
    if not dataset:
        raise NotImplementedError('dataset for {} task is not '
                                'implemented.'.format(task_type))
    print_rank_0(' > {} found {} samples.'.format(task, len(dataset)))
    return dataset


def _get_data_path(task):
    args = get_args()
    valid_data_path = os.path.join(args.valid_data[0], f"{task}.jsonl")
    assert os.path.exists(valid_data_path), f'Vaild data file {valid_data_path} is not exist.'
    return valid_data_path



class BaseQA(torch.utils.data.Dataset):
    def __init__(self, path, template):
        self.examples = []
        self.dataset = []
        self.options_string = 'ABCDEFGH'
        self.template = template
        with open(path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                self.dataset.append(data)
        self.class_num = len(self.dataset[0]['options'])
        self.preprocess_dataset()
    
    def preprocess_example(self, example):
        input_str = self.template.format(
            passage=example['passage'],
            question=example['question']
        )
        answer_str = []
        if len(example["options"]) != self.class_num:
            return None, None, None
        if len(example["label"]) > 1 or example["label"] not in self.options_string:
            print_rank_0(f"example: {example}")
            return None, None, None
        for i in range(self.class_num):
            answer_str.append(' ' + self.replace_options(example["options"][i]))
        label = self.options_string.index(example["label"])
        return input_str, answer_str, label

    def preprocess_dataset(self):
        for example in self.dataset:
            example = self.preprocess_example(example)
            if example[0] is None:
                continue
            self.examples.append(example)
    
    def replace_options(self, option):
        op_list = [f'({o})' for o in self.options_string[:self.class_num]]
        for op in op_list:
            option = option.replace(op, '')
        return option

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_str, output_str, label = self.examples[index]
        return input_str, output_str, label



class _ChineseQADataset(BaseQA):
    def __init__(self, path):
        template = "{passage}\n问题: {question}\n答案:"
        super().__init__(path, template)
    
    
class _EnglishQADataset(BaseQA):
    def __init__(self, path):
        template = "{passage}\nQueation: {question}\nAnswer:"
        super().__init__(path, template)


class BaseCloze(torch.utils.data.Dataset):
    def __init__(self, path):
        args = get_args()
        self.seq_len = args.seq_length
        self.dataset = []
        self.path= path
        self.tokenizer = get_tokenizer()
        self.tokenizer.tokenizer.pad_token = self.tokenizer.tokenizer.eos_token
        self.pad_idx = self.tokenizer.pad
        self.preprocess_dataset()
    
    def preprocess_dataset(self):
        raise NotImplemented("Preprocess Dataset Not Implemeted.")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenizer.encode(
            self.dataset[idx][0])
        num_tokens = len(tokens)
        pad_mask = [0] * num_tokens
        labels = self.tokenizer.tokenizer.encode(
            self.dataset[idx][1])
        pad_mask += [1] * len(labels)
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        else:
            tokens = tokens[len(tokens)-self.seq_len: ]  # keep answer
        pad_mask = np.array(pad_mask[1:])

        return {'text': np.array(tokens), 'pad_mask': pad_mask}


class _ChineseClozeDataset(BaseCloze):
    def __init__(self, path):
        super().__init__(path)
    
    def preprocess_dataset(self):
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                question = f"问题: {data['question']}\n答案:"
                answer = f" {data['answer']}"
                self.dataset.append((question, answer))

class _EnglishClozeDataset(BaseCloze):
    def __init__(self, path):
        super().__init__(path)
    
    def preprocess_dataset(self):
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                question = f"Question: {data['question']}\nAnswer:"
                answer = f" {data['answer']}"
                self.dataset.append((question, answer))
