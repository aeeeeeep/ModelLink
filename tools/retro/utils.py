# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/NVIDIA/Megatron-LM/blob/main/tools/retro/utils.py
# reworked/refactored some parts to make it run.

import os
import types
import torch

from megatron import get_retro_args
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
    _GPTSentencePieceTokenizer,
)


def get_args_path(workdir):
    '''Argument copy stored within retro workdir.'''
    return os.path.join(workdir, "args.json")


def get_num_chunks_per_sample():
    '''Compute seq_length // chunk_length.'''
    args = get_retro_args()
    sample_length = args.retro_gpt_seq_length
    chunk_length = args.retro_gpt_chunk_length
    if sample_length % chunk_length != 0:
        raise ValueError('chunk_length should be divisible by sample_length.')
    return sample_length // chunk_length


def get_gpt_tokenizer():
    '''GPT (BPE) tokenizer.'''
    args = get_retro_args()
    tokenizer_type = args.retro_gpt_tokenizer_type
    if tokenizer_type == "GPT2BPETokenizer":
        if not args.retro_gpt_vocab_file or not args.retro_gpt_merge_file:
            raise ValueError('retro_gpt_vocab_file and retro_gpt_merge_file should not be none.')
        return _GPT2BPETokenizer(
            vocab_file=args.retro_gpt_vocab_file,
            merge_file=args.retro_gpt_merge_file,
        )
    elif tokenizer_type == 'GPTSentencePieceTokenizer':
        if args.retro_gpt_tokenizer_model is None:
            raise ValueError('retro_gpt_tokenizer_model is None.')
        return _GPTSentencePieceTokenizer(args.retro_gpt_tokenizer_model)
    else:
        raise Exception("unrecognized gpt tokenizer, '%s'." % tokenizer_type)


def get_bert_tokenizer():
    '''Bert (Wordpiece) tokenizer.'''
    args = get_retro_args()
    lower_case = {
        "BertWordPieceLowerCase" : True,
        "BertWordPieceCase" : False,
    }[args.retro_bert_tokenizer_type]
    return _BertWordPieceTokenizer(
        vocab_file=args.retro_bert_vocab_file,
        lower_case=lower_case,
    )


class GPTToTextDataset(torch.utils.data.Dataset):
    '''Dataset to convert GPT tokens to text.'''

    def __init__(self, gpt_dataset):

        super().__init__()

        self.gpt_dataset = gpt_dataset
        self.gpt_tokenizer = get_gpt_tokenizer()

    def __len__(self):
        return len(self.gpt_dataset)

    def __getitem__(self, idx):
        gpt_token_ids = self.gpt_dataset[idx]["text"].tolist()
        text = self.gpt_tokenizer.detokenize(gpt_token_ids)
        return {"text": text}
