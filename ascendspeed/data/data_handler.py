import os
import sys
import time
import glob
import json
from typing import Tuple

import torch
import numpy as np
from datasets import load_dataset

from ascendspeed.data import indexed_dataset
from ascendspeed.data.prompter import Prompter, ALPACA_TEMPLATE


__all__ = ["get_dataset_handler", "build_dataset"]


DEFAULT_CACHE_DIR = "~/tmp"


class BaseDatasetHandler(object):
    """
    a base handler to tokenize or/and prompt your own dataset
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        self.args = args
        self.tokenizer = tokenizer
        self.splitter = splitter
        self.raw_datasets = raw_datasets
        self.max_seq_len = args.seq_length
        self.tokenized_dataset = None

    @property
    def _unwrapped_tokenizer(self):
        """get huggingface tokenizer"""
        return self.tokenizer.tokenizer

    def _tokenize(self, prompt):
        result = self._unwrapped_tokenizer(text=prompt)
        result["labels"] = result["input_ids"].copy()

        return result

    def _filter(self, sample):
        """prompt and tokenize"""
        return NotImplemented

    def get_tokenized_data(self):
        """return tokenized(and prompted) data"""
        columns = next(iter(self.raw_datasets)).keys()
        remove_columns = list(set(columns) - set(self.args.json_keys))
        return self.raw_datasets.map(self._filter,
                                     remove_columns=remove_columns,
                                     num_proc=self.args.workers)

    def serialize_to_disk(self):
        """save idx and bin to disk"""
        startup_start = time.time()
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.get_tokenized_data()
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        print(f"Vocab size: {self.tokenizer.vocab_size}")
        print(f"Output prefix: {self.args.output_prefix}")
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                         impl=self.args.dataset_impl,
                                                         vocab_size=None)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, doc in enumerate(iter(self.tokenized_dataset), start=1):
            for key in self.args.json_keys:
                sentences = doc[key]
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    # TODO: not access to _dtype attribute
                    total_bytes_processed += len(sentence) * builders[key]._dtype().itemsize
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if i % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                    f"({i/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])


class GeneralPretrainHandler(BaseDatasetHandler):
    """
    a general pretrain dataset handler
    """
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        if self._text_keys:
            self.args.json_keys = self._text_keys

    @property
    def _text_keys(self):
        return []

    def _filter(self, sample):
        for key in self.args.json_keys:
            text = sample[key]
            doc_ids = []
            for sentence in self.splitter.tokenize(text):
                if len(sentence) > 0:
                    sentence_ids = self._tokenize(sentence)
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1]['input_ids'].append(self.tokenizer.eod)
                doc_ids[-1]['attention_mask'].append(1)
                doc_ids[-1]['labels'].append(self.tokenizer.eod)
            sample[key] = doc_ids
            # for now, only input_ids are saved
            sample[key] = list(map(lambda x: x['input_ids'], sample[key]))
        return sample


class GeneralInstructionHandler(BaseDatasetHandler):
    """
    a general instruction dataset handler
    """
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = Prompter(ALPACA_TEMPLATE)
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100

    @property
    def _instruction_key(self) -> str:
        return "instruction"

    @property
    def _input_key(self) -> str:
        return "input"

    @property
    def _output_key(self) -> str:
        return "output"

    @property
    def _human_prefix(self) -> str:
        return None

    @property
    def _assistant_prefix(self) -> str:
        return None

    def _prompt(self, sample):
        """format sample info"""
        if self._human_prefix:
            messages = [
                dict(role="user", content=sample["instruction"] + "\n" + sample["input"]),
                dict(role="assistant", content=sample["output"]),
            ]
        else:
            messages = []
            turns = sample["instruction"].split("Human:")

            for val in turns:
                if val:
                    tmp = val.split("Assistant:")
                    if len(tmp) > 1:
                        messages.append(dict(role="user", content=tmp[0].strip()))
                        messages.append(dict(role="assistant", content=tmp[1].strip()))
                    else:
                        messages.append(dict(role="assistant", content=tmp[0].strip()))
            messages.pop()
            messages.append(dict(role="assistant", content=sample["output"].strip()))
        return messages

    def _filter(self, sample):
        messages = self._prompt(sample)

        full_prompt = self.prompter.generate_training_prompt(messages)

        tokenized_full_prompt = self._tokenize(full_prompt)

        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        if not self.train_on_inputs:
            user_prompt = full_prompt.rsplit(self.prompter.template["assistant_token"], 
                                             maxsplit=1)[0] + \
                self.prompter.template["assistant_token"] + "\n"
            tokenized_user_prompt = self._tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [self.ignored_label] * user_prompt_len + \
                tokenized_full_prompt["labels"][user_prompt_len:]

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]

        return tokenized_full_prompt


class BelleMultiTurnInstructionHandler(GeneralInstructionHandler):
    """
    BelleMultiTurn dataset handler
    """
    @property
    def _human_prefix(self) -> str:
        return "Human:"

    @property
    def _assistant_prefix(self) -> str:
        return "Assistant:"


def _get_handler_cls(handler_name=None):
    """choose dataset class by dataset_name"""
    current_module = sys.modules[__name__]
    handler = getattr(current_module, handler_name, None)
    if handler is None:
        handler = GeneralPretrainHandler
    print(f"dataset will use {handler.__name__} to handle dataset")
    return handler


def get_dataset_handler(args, raw_dataset, tokenizer, splitter):
    """
    get a handler instance
    """
    handler = _get_handler_cls(args.handler_name)

    handler_instance = handler(args, raw_dataset, tokenizer, splitter)
    return handler_instance


def _get_data_format(files):
    """get format with largest number"""
    all_support_format = {
        'parquet': 'parquet',
        'arrow': 'arrow',
        'csv': 'csv',
        'json': 'json',
        'jsonl': 'json',
        'txt': 'text'
    }
    format_num = {}
    for file in files:
        ext = file.split('.')[-1]
        format_num[ext] = format_num.get(ext, 0) + 1
    exts_with_num = sorted(format_num.items(), key=lambda x: x[1], reverse=True)
    has_data_file = False
    for ext, _ in exts_with_num:
        if ext in all_support_format:
            has_data_file = True
            break
    return (ext, all_support_format[ext]) if has_data_file else (None, None)


def _has_py_script(input_name):
    if os.path.isdir(input_name):
        dir_name = os.path.basename(input_name)
        if os.path.exists(os.path.join(input_name, dir_name + '.py')):
            has_py_script = True
        else:
            has_py_script = False
    else:
        if input_name.split('.')[-1] == 'py':
            has_py_script = True
        else:
            has_py_script = False
    return has_py_script


def build_dataset(args):
    """loading dataset by huggingface"""
    if args.hf_datasets_params:
        with open(args.hf_datasets_params, 'r') as fin:
            param_dict = json.load(fin)
        return load_dataset(**param_dict)
    cache_dir = DEFAULT_CACHE_DIR
    load_from_local = os.path.exists(args.input)
    if load_from_local:
        if _has_py_script(args.input):
            print("loading data from a local python script")
            raw_datasets = load_dataset(
                args.input,
                split="train",
                num_proc=None if args.streaming else args.workers,
                cache_dir=cache_dir,
                streaming=args.streaming
            )
        else:
            data_files = [args.input] if os.path.isfile(args.input) else \
                glob.glob(os.path.join(args.input, '*'))
            ext, data_format = _get_data_format(data_files)
            filtered_data_files = list(filter(lambda x: x.split('.')[-1] == ext, data_files))
            if filtered_data_files:
                print(f"loading data from local file, format: {data_format},"
                      f" file num: {len(data_files)}")
                raw_datasets = load_dataset(
                    data_format,
                    split="train",
                    data_files=filtered_data_files,
                    num_proc=None if args.streaming else args.workers,
                    cache_dir=cache_dir,
                    streaming=args.streaming
                )
            else:
                raise Exception("unknown local data!")
    else:
        print("loading data from remote huggingface")
        raw_datasets = load_dataset(
            args.input,
            split="train",
            num_proc=None if args.streaming else args.workers,
            cache_dir=cache_dir,
            streaming=args.streaming
        )
    return raw_datasets
