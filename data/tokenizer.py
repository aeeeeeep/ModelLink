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

"""Megatron tokenizers. just using huggingface implementation."""

from abc import ABC
from abc import abstractmethod
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import BertTokenizer as FullBertTokenizer
from megatron.tokenizer.tokenizer import _BertWordPieceTokenizer, _GPT2BPETokenizer
from data.utils import ensure_var_is_not_none, ensure_valid


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == "PretrainedFromHF":
        ensure_var_is_not_none(args.tokenizer_name_or_path)

        # prevent transformers from logging info and warnings on each rank
        import transformers
        import logging
        if args.rank == 0:
            transformers.utils.logging.set_verbosity(logging.INFO)
        else:
            # shut the warnings on replicas
            transformers.utils.logging.set_verbosity(logging.ERROR)

        if args.rank == 0:
            print(" vocab file is un-used. loading tokenizer from pre-trained model")
        tokenizer = _AutoTokenizer(
            args.tokenizer_name_or_path,
            vocab_extra_ids=args.vocab_extra_ids,
            model_max_length=args.seq_length,
            use_fast=args.tokenizer_not_use_fast)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                      args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Apply the requested rules to change the size of the vocabulary"""
    if args.pad_vocab_size_to is not None:
        if args.pad_vocab_size_to < orig_vocab_size:
            raise ValueError(
                f"You asked to pad the vocabulary to {args.pad_vocab_size_to} when the initial vocabulary size is "
                f"{orig_vocab_size}. You can only pad to a higher value."
            )

        if args.make_vocab_size_divisible_by is not None and \
            (args.pad_vocab_size_to % args.make_vocab_size_divisible_by) != 0:
            raise ValueError(f"{args.pad_vocab_size_to} is not divisible by {args.make_vocab_size_divisible_by}")

        after = args.pad_vocab_size_to
    else:
        # Pad vocab size so it is divisible by model parallel size and still having GPU friendly size.
        after = orig_vocab_size
        multiple = args.make_vocab_size_divisible_by * \
            args.tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class _AutoTokenizer(AbstractTokenizer):
    """AutoTokenizer for Hf Pretrained model loading."""

    def __init__(self, tokenizer_name_or_path, vocab_extra_ids, model_max_length, use_fast):
        name = tokenizer_name_or_path
        super().__init__(name)
        hf_tokenizer_kwargs = {}
        if vocab_extra_ids > 0:
            hf_tokenizer_kwargs["additional_special_tokens"] = [f"<extra_id_{_id}>" for _id in range(vocab_extra_ids)]
       
        hf_tokenizer_kwargs["model_max_length"] = model_max_length
        hf_tokenizer_kwargs["use_fast"] = use_fast
        hf_tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **hf_tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer) # vocab_size doesn't contain additional tokens

    @property
    def vocab(self):
        return {
            **{special_token: self.tokenizer.convert_tokens_to_ids(special_token) 
            for special_token in self.tokenizer.additional_special_tokens},
            **self.tokenizer.vocab,
        }

    @property
    def inv_vocab(self):
        return {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eos

    @property
    def cls(self):
        candidate = self.tokenizer.cls_token_id
        return self._check_token_candidate(candidate)

    @property
    def sep(self):
        candidate = self.tokenizer.sep_token_id
        return self._check_token_candidate(candidate)

    @property
    def pad(self):
        candidate = self.tokenizer.pad_token_id

        # just use eos_token_id if pad_token_id is not available, it is reasonable
        # maybe add a new token, and resize embedding layer is better
        if candidate is None:
            candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def mask(self):
        candidate = self.tokenizer.mask_token_id
        return self._check_token_candidate(candidate)

    @property
    def bos(self):
        raise NotImplementedError("Missing <bos>")

    @property
    def eos(self):
        candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def additional_special_tokens_ids(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self.tokenizer.additional_special_tokens_ids

    @staticmethod
    def _check_token_candidate(candidate):
        if candidate is None:
            raise AttributeError("Token doesn't exist")
        return candidate
