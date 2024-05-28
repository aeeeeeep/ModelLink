# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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
import sys
import time
import glob
import json
import logging
from typing import List

import torch
import inspect
import numpy as np
from functools import partial
from datasets import load_dataset

from megatron.core.datasets import indexed_dataset
from modellink.data.prompter import Prompter, AlpacaTemplate, LfAlpacaTemplate
from modellink.data.templates import get_templates, Role, get_template_and_fix_tokenizer
from modellink.data.parser import DatasetAttr
from typing import Any, Dict, List, Optional
from datasets import concatenate_datasets, interleave_datasets, Features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["get_dataset_handler", "build_dataset"]

DEFAULT_CACHE_DIR = "~/tmp"
DATA_CONFIG = "dataset_info.json"
FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


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

    def get_tokenized_data(self):
        """get tokenized(and prompted) data"""
        columns = next(iter(self.raw_datasets)).keys()
        remove_columns = list(set(columns) - set(self.args.json_keys))
        proc_kwargs = {} if self.args.streaming else {"num_proc": self.args.workers}
        return self.raw_datasets.map(self._filter, remove_columns=remove_columns, **proc_kwargs)

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

        logger.info("Vocab size: %s", self.tokenizer.vocab_size)
        logger.info("Output prefix: %s", self.args.output_prefix)
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(output_bin_files[key])
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        logger.info("Time to startup:%s", startup_end - startup_start)

        skip_num = 0
        for i, doc in enumerate(iter(self.tokenized_dataset), start=1):
            for key in self.args.json_keys:
                sentences = doc[key]
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    if self.args.seq_length is not None and len(sentence) >= self.args.seq_length:
                        skip_num += 1
                        continue

                    total_bytes_processed += len(sentence) * np.int32().itemsize
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if i % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                logger.info("Processed %s documents (%s docs/s, %s MB/s).", i, i / elapsed, mbs)

        logger.info("Skip %s sample exceeded seq-length(%s)", skip_num // 3, self.args.seq_length)
        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def _tokenize(self, prompt):
        result = self._unwrapped_tokenizer(text=prompt)
        result["labels"] = result["input_ids"].copy()

        return result

    def _filter(self, sample):
        """prompt and tokenize"""
        return NotImplemented


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

    def _pre_process(self, sample):
        return sample

    def _filter(self, sample):
        sample = self._pre_process(sample)
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


class AlpacaPretrainHandler(GeneralPretrainHandler):
    """
    alpaca-data-conversation pretrain dataset handler
    """
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
       
        self.message_format = "A chat between a curious user and an artificial intelligence assistant. " \
                              "The assistant gives helpful, detailed, and polite answers to the user's questions." \
                              "USER: Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n" \
                              "### Instruction:\n{instruction}\n\n###{inputs}\n\n### Response: ASSISTANT: {response}"

    def _filter(self, sample):
        key = "text"
        text = self.message_format.format(
            instruction=sample.get("instruction"), 
            inputs=f" Input:\n{sample.get('input')}" if sample.get("input") else None,
            response=sample.get("output"))
        doc_ids = []
        for sentence in self.splitter.tokenize(text):
            if len(sentence) > 0:
                sentence_ids = self._tokenize(sentence)
                doc_ids.append(sentence_ids)
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids[-1]['input_ids'].append(self.tokenizer.eod)
        sample[key] = doc_ids
        sample[key] = list(map(lambda x: x['input_ids'], sample[key]))
        return sample

def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

# LlamaFactory通用格式
class LlamaFactoryInstructionHandler(BaseDatasetHandler):
    """
    a Llama-factory Alpaca instruction dataset handler
    """
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        # self.prompter is unused in LlamaFactoryInstructionHandler
        self.prompter = Prompter(LfAlpacaTemplate())
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = self._is_muti_turn()
        self.dataset_attr = get_dataset_list(args)
        self.llama_factory_template = get_template_and_fix_tokenizer(tokenizer.tokenizer, args.template)


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
    def _history_key(self)-> str:
        return "history"

    @property
    def _system(self)-> str:
        return "system"

    @property
    def _human_prefix(self) -> str:
        return "Human:"

    @property
    def _assistant_prefix(self) -> str:
        return "Assistant:"
    
    def _is_muti_turn(self) -> bool:
        try:
            is_multi_turn = True if isinstance(self._history_key, str) else False
        except NotImplementedError:
            is_multi_turn = False
        return is_multi_turn

    def _format_msg(self, sample):
        return sample

    # ### convert_alpaca
    # def _format_msg(self, sample):
    #     """format sample info"""
    #     outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    #     prompt = []
        
    #     if self._history_key in sample.keys() and isinstance(sample[self._history_key], dict):
    #         for old_prompt, old_response in sample[self._history_key]:
    #             prompt.append({"role": Role.USER.value, "content": old_prompt})
    #             prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    #     content = []
    #     if self._instruction_key and sample[self._instruction_key]:
    #         content.append(sample[self._instruction_key])

    #     if self._input_key and sample[self._input_key]:
    #         content.append(sample[self._input_key])

    #     prompt.append({"role": Role.USER.value, "content": "\n".join(content)})

    #     if self._output_key and isinstance(sample[self._output_key], list):
    #         response = [
    #             {"role": Role.ASSISTANT.value, "content": content} for content in examples[self._output_key]
    #         ]
    #     elif self._output_key and isinstance(sample[self._output_key], str):
    #         response = [{"role": Role.ASSISTANT.value, "content": sample[self._output_key]}]
    #     else:
    #         response = []

    #     outputs["prompt"]=prompt
    #     outputs["response"]=response
    #     outputs["system"].append(sample[self._system] if self._system in sample.keys() and self._system else "")
    #     outputs["tools"].append("")
    #     return outputs

    def _tokenize_prompt(
        self,
        example: Dict[str, List[Any]],
        template: "Template",
        tokenizer: "PreTrainedTokenizer",
) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        # if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
        #     return model_inputs
        # example["prompt"]):[{'role': 'user', 'content': 'Summarize the given news article in one sentence.\nIn the latest budget announcement, the Chancellor has promised to introduce a new employment allowance to help employees of small businesses.'}, {'role': 'assistant', 'content': 'The government has announced an employment allowance to help employees of small businesses in their most recent budget announcement.'}]
        if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
            # this message is unvalid
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
        else:
            messages = example["prompt"] + example["response"]

        for source_ids, target_ids in self.llama_factory_template.encode_multiturn(
            tokenizer, messages, example["system"][0], example["tools"][0]
        ):
            ### todo train_on_inputs这个参数作用是什么
            if self.train_on_inputs: 
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [self.ignored_label] * (len(source_ids) - 1)
            else:
                source_mask = [self.ignored_label] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        total_length = len(input_ids)
        # block_size = 8
        # # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        # total_length = (total_length // block_size) * block_size
        # # split by chunks of cutoff_len
        # for i in range(0, total_length, block_size):
        #     if not all(label == self.ignored_label for label in labels[i : i + block_size]):
        #         model_inputs["input_ids"].append(input_ids[i : i + block_size])
        #         model_inputs["attention_mask"].append([1] * block_size)
        #         model_inputs["labels"].append(labels[i : i + block_size])
        model_inputs["input_ids"] = input_ids
        # "input_ids[0] // input_ids[0]"" is to ensure that the dtype of attention_mask is consistent with the input_ids
        ### Todo if input_ids[0] == 0,it will raise exception. have better method?
        model_inputs["attention_mask"] = [input_ids[0] // input_ids[0]] * total_length # todo
        model_inputs["labels"] = labels
        return model_inputs

    def _filter(self, sample):
        messages = self._format_msg(sample)
        tokenized_full_prompt = self._tokenize_prompt(messages, self.llama_factory_template, self.tokenizer.tokenizer)

        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
        return tokenized_full_prompt

class GeneralInstructionHandler(BaseDatasetHandler):
    """
    a general instruction dataset handler
    """
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = Prompter(AlpacaTemplate())
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = self._is_muti_turn()

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
        raise NotImplementedError

    @property
    def _assistant_prefix(self) -> str:
        raise NotImplementedError
    
    def _is_muti_turn(self) -> bool:
        try:
            is_multi_turn = True if isinstance(self._human_prefix, str) else False
        except NotImplementedError:
            is_multi_turn = False
        return is_multi_turn

    def _format_msg(self, sample):
        """format sample info"""
        if not self.is_multi_turn:
            messages = [
                dict(
                    role=self.prompter.user_role,
                    content=sample[self._instruction_key] + "\n" + sample[self._input_key]),
                dict(role=self.prompter.assistant_role, content=sample[self._output_key])
            ]
            return messages
        
        messages = []
        turns = sample[self._instruction_key].split(self._human_prefix)

        for msg in turns:
            if not msg:
                continue
            tmp = msg.split(self._assistant_prefix)
            if len(tmp) > 1:
                messages.append(dict(role=self.prompter.user_role, content=tmp[0].strip()))
                messages.append(dict(role=self.prompter.assistant_role, content=tmp[1].strip()))
            else:
                messages.append(dict(role=self.prompter.assistant_role, content=tmp[0].strip()))
        messages.pop()
        messages.append(dict(role=self.prompter.assistant_role, content=sample[self._output_key].strip()))
        return messages

    def _filter(self, sample):
        messages = self._format_msg(sample)
        full_prompt = self.prompter.generate_training_prompt(messages)
        tokenized_full_prompt = self._tokenize(full_prompt)

        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        if not self.train_on_inputs:
            user_prompt = full_prompt.rsplit(self.prompter.template.assistant_token, maxsplit=1)[0] + \
                self.prompter.template.assistant_token + "\n"
            tokenized_user_prompt = self._tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"][:user_prompt_len] = [self.ignored_label] * user_prompt_len

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
        print("tokenized_full_prompt")
        print(tokenized_full_prompt)
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


class MOSSMultiTurnHandler(GeneralInstructionHandler):
    
    @property
    def user_token(self) -> List[int]:
        #Apply for baichuan
        return [195]

    @property
    def assistant_token(self) -> List[int]:
        return [196]

    @property
    def ignored_index(self) -> List[int]:
        return [-100]

    def _filter(self, sample):
        input_ids, labels = [], []
        for turn in sample["chat"].values():
            if not turn:
                continue

            user = turn["Human"].replace("<eoh>", "").replace("<|Human|>: ", "").strip()
            assistant = turn["MOSS"].replace("<|MOSS|>:", "").replace("<eom>", "").strip()

            user_ids = self._unwrapped_tokenizer.encode(user)
            assistant_ids = self._unwrapped_tokenizer.encode(assistant)

            input_ids += self.user_token + user_ids + self.assistant_token + assistant_ids
            labels += [self._unwrapped_tokenizer.eos_token_id] + self.ignored_index * len(
                user_ids) + self.ignored_index + assistant_ids
                
        input_ids.append(self._unwrapped_tokenizer.eos_token_id)
        labels.append(self._unwrapped_tokenizer.eos_token_id)
        attention_mask = [1 for _ in range(len(input_ids))]

        return {
            "input_ids" : [input_ids],
            "attention_mask" : [attention_mask],
            "labels" : [labels]
        }


class MOSSInstructionHandler(GeneralInstructionHandler):
    def _filter(self, sample):
        messages = []
        tokenized_chats = []

        for turn in sample["chat"].values():
            if not turn:
                continue

            user = turn["Human"].replace("<eoh>", "").replace("<|Human|>: ", "").strip()
            assistant = turn["MOSS"].replace("<|MOSS|>:", "").replace("<eom>", "").strip()

            messages.append(dict(role=self.prompter.user_role, content=user))
            messages.append(dict(role=self.prompter.assistant_role, content=assistant))

            full_prompt = self.prompter.generate_training_prompt(messages)
            tokenized_full_prompt = self._tokenize(full_prompt)

            if not self.train_on_inputs:
                user_prompt = full_prompt.rsplit(self.prompter.template.assistant_token, maxsplit=1)[0] + \
                              self.prompter.template.assistant_token + "\n"
                tokenized_user_prompt = self._tokenize(user_prompt)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                             user_prompt_len:]

            tokenized_chats.append(tokenized_full_prompt)

        for key in self.args.json_keys:
            sample[key] = [chat[key] for chat in tokenized_chats]

        return sample


class LeetcodePythonInstructionHandler(GeneralInstructionHandler):
    @property
    def _instruction_key(self) -> str:
        return "code_with_problem"

    @property
    def _input_key(self) -> str:
        return "code_only"

    @property
    def _output_key(self) -> str:
        return "explanation_only"

    def _format_msg(self, sample):
        """format sample info"""
        messages = [
            dict(
                role=self.prompter.user_role,
                content=sample[self._instruction_key].split("```", maxsplit=1)[0].strip()),
            dict(
                role=self.prompter.assistant_role,
                content=sample[self._input_key] + "\n" + sample[self._output_key])
        ]
        return messages


class StackOverflowPythonPretrainHandler(GeneralPretrainHandler):
    @property
    def _text_keys(self):
        return ['text']

    def _pre_process(self, sample):
        sample['text'] = f"In python, {sample['title']}\n### Question:\n{sample['question_body']}\n" \
                         f"### Response:\n{sample['answer_body']}\n"


def _get_handler_cls(handler_name=None):
    """choose dataset class by dataset_name"""
    current_module = sys.modules.get(__name__)
    if not current_module:
        raise Exception("curent module not found")
    handler = getattr(current_module, handler_name, None)
    if handler is None:
        handler = GeneralPretrainHandler
    logger.info("dataset will use %s to handle dataset", handler.__name__)
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
    return (ext, all_support_format.get(ext)) if has_data_file else (None, None)


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



def use_modelscope() -> bool:
    return bool(int(os.environ.get("USE_MODELSCOPE_HUB", "0")))

def get_dataset_list(data_args) -> List["DatasetAttr"]:
    """Map multiple dataset attributes to List["DatasetAttr"]
    through parameters and the data.json mapping file."""
    if data_args.input is not None:
        dataset_names = [ds.split("/")[-1].strip() for ds in data_args.input.split(",")]
    else:
        dataset_names = []

    if data_args.dataset_dir == "ONLINE":
        dataset_info = None
    else:
        try:
            with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
                dataset_info = json.load(f)
        except Exception as err:
            if len(dataset_names) != 0:
                raise ValueError(
                    "Cannot open {} due to {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG), str(err))
                )
            dataset_info = None
    ### Multiple Dataset Interleaving Probability
    if data_args.interleave_probs is not None:
        data_args.interleave_probs = [float(prob.strip()) for prob in data_args.interleave_probs.split(",")]

    dataset_list: List[DatasetAttr] = []
    for name in dataset_names:
        if dataset_info is None:
            load_from = "ms_hub" if use_modelscope() else "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url:
            if (use_modelscope() and has_ms_url) or (not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"])
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        else:
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")

        if "columns" in dataset_info[name]:
            column_names = ["system", "images"]
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history"])
            else:
                column_names.extend(["messages", "tools"])

            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
            tag_names = (
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            )
            for tag in tag_names:
                dataset_attr.set_attr(tag, dataset_info[name]["tags"])

        dataset_list.append(dataset_attr)

    return dataset_list

def convert_alpaca(sample: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"):
    """format sample info
    {
      "instruction": "我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？",
      "input": "",
      "output": "中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。",
      "history": [
       [
        "回答的非常好",
        "感谢你的认可！还有什么需要我帮助的吗？"
       ]
      ]
     }
    ---->>>>
    {
        'prompt': [{'role': 'user', 'content': '回答的非常好'}, 
                {'role': 'assistant', 'content': '感谢你的认可！还有什么需要我帮助的吗？'}, 
                {'role': 'user', 'content': '我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？'}], 
        'response': [{'role': 'assistant', 'content': '中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。'}], 
        'system': [''], 
        'tools': ['']
    }
     """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    
    if dataset_attr.history and isinstance(sample[dataset_attr.history], dict):
        for old_prompt, old_response in sample[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    content = []
    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        content.append(sample[dataset_attr.prompt])

    if dataset_attr.query and sample[dataset_attr.query]:
        content.append(sample[dataset_attr.query])

    prompt.append({"role": Role.USER.value, "content": "\n".join(content)})

    if dataset_attr.response and isinstance(sample[dataset_attr.response], list):
        response = [
            {"role": Role.ASSISTANT.value, "content": content} for content in sample[dataset_attr.response]
        ]
    elif dataset_attr.response and isinstance(sample[dataset_attr.response], str):
        response = [{"role": Role.ASSISTANT.value, "content": sample[dataset_attr.response]}]
    else:
        response = []

    outputs["prompt"]=prompt
    outputs["response"]=response
    outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
    outputs["tools"].append("")
    return outputs


# todo 待开发
def convert_sharegpt(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}

    return outputs


def align_dataset(dataset, dataset_attr, data_args):
    r"""
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}]
        system: "..."
        tools: "...",
        images: [],
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)
    column_names = list(next(iter(dataset)).keys())
    # features = Features.from_dict(
    #     {
    #         "prompt": [
    #             {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
    #         ],
    #         "response": [
    #             {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
    #         ],
    #         "system": [{"dtype": "string", "_type": "Value"}],
    #         "tools": [{"dtype": "string", "_type": "Value"}],
    #     }
    # )
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.workers,
            load_from_cache_file=(not data_args.overwrite_cache),
            desc="Converting format of dataset",
        )

    return dataset.map(
        convert_func,
        remove_columns=column_names,
        **kwargs,
    )

def load_single_dataset(dataset_attr, data_args):
    """loading single dataset by huggingface/modelscope/script/local file"""
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File not found.")

        if data_path is None:
            raise ValueError("File extension must be txt, csv, json or jsonl.")
    else:
        raise NotImplementedError

    if dataset_attr.load_from == "ms_hub":
        try:
            from modelscope import MsDataset
            from modelscope.utils.config_ds import MS_DATASETS_CACHE

            cache_dir = data_args.cache_dir or MS_DATASETS_CACHE
            dataset = MsDataset.load(
                dataset_name=data_path,
                subset_name=data_name,
                data_dir=data_dir,
                data_files=data_files,
                split=data_args.split,
                cache_dir=cache_dir,
                token=data_args.ms_hub_token,
                use_streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
            )
            if isinstance(dataset, MsDataset):
                dataset = dataset.to_hf_dataset()
        except ImportError:
            raise ImportError("Please install modelscope via `pip install modelscope -U`")
    else:
        if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
            kwargs = {"trust_remote_code": True}
        else:
            kwargs = {}
        cache_dir = data_args.cache_dir or DEFAULT_CACHE_DIR
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=data_args.split,
            cache_dir=cache_dir,
            token=data_args.hf_hub_token,
            streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
            **kwargs,
        )

    # if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
    #     dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

    if data_args.max_samples is not None:  # truncate dataset
        num_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

    return align_dataset(dataset, dataset_attr, data_args)

def merge_dataset(all_datasets, data_args):
    """Merging multiple Datasets by mix_strategy"""
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=data_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy.")

def build_dataset(args):
    """loading dataset by huggingface"""
    raw_datasets = None
    if (args.handler_name == "LlamaFactoryInstructionHandler"):
        all_datasets = []
        for dataset_attr in get_dataset_list(args):
            all_datasets.append(load_single_dataset(dataset_attr, args))
        raw_datasets = merge_dataset(all_datasets, args)
    else:
        if args.handler_name == "MOSSInstructionHandler" or args.handler_name == "MOSSMultiTurnHandler":
            # for MOSS, streaming is needed.
            args.streaming = True
        if args.hf_datasets_params:
            with open(args.hf_datasets_params, 'r') as fin:
                param_dict = json.load(fin)
            return load_dataset(**param_dict)
        cache_dir = DEFAULT_CACHE_DIR
        split_flag = "train"
        load_from_local = os.path.exists(args.input)
        if load_from_local:
            if _has_py_script(args.input):
                logger.info("loading data from a local python script")
                raw_datasets = load_dataset(
                    args.input,
                    split=split_flag,
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
                    logger.info("loading data from local file, format: %s," 
                                " file num: %s", data_format, len(data_files))
                    raw_datasets = load_dataset(
                        data_format,
                        split=split_flag,
                        data_files=filtered_data_files,
                        num_proc=None if args.streaming else args.workers,
                        cache_dir=cache_dir,
                        streaming=args.streaming
                    )
                else:
                    raise Exception("unknown local data!")
        else:
            logger.info("loading data from remote huggingface")
            raw_datasets = load_dataset(
                args.input,
                split=split_flag,
                num_proc=None if args.streaming else args.workers,
                cache_dir=cache_dir,
                streaming=args.streaming
            )
        if raw_datasets == None:
            raise Exception("unknown data!")

    return raw_datasets
