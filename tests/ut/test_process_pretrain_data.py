import unittest
import sys
import os
import math

from utils import judge_expression
import modellink

from modellink.tokenizer import build_tokenizer
from modellink.tokenizer.tokenizer import _AutoTokenizer
from modellink.data.data_handler import GeneralPretrainHandler
from modellink.data.data_handler import build_dataset, get_dataset_handler
from tools.preprocess_data import get_args, build_splitter


class TestProcessPretrainData:

    def setup_class(self):
        sys.argv = [
            sys.argv[0],
            "--input", "/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
            "--tokenizer-type", "PretrainedFromHF",
            "--output-prefix", "/data/pretrain_dataset/alpaca",
            "--tokenizer-name-or-path", "/data/llama-2-7b-hf",
            "--workers", "4",
            "--log-interval", "1000"
        ]
        self.args = get_args()
        self.tokenizer = build_tokenizer(self.args)
        self.splitter = build_splitter(self.args)
        self.raw_dataset = build_dataset(self.args)
        self.handler = get_dataset_handler(self.args, self.raw_dataset, self.tokenizer, self.splitter)
    
    def test_build_tokenizer(self):
        """
        Test normal function of the tokenizer:
            the instance of tokenizer
            the length of vocabulary
            the encode function
            the decode function
            the eos append
            ...(If missed something else, welcome to add)
        """
        judge_expression(isinstance(self.tokenizer, _AutoTokenizer))
        judge_expression(self.tokenizer.vocab_size == 32000)
        judge_expression(self.tokenizer.tokenize('bug') == [1, 6494])
        judge_expression(self.tokenizer.detokenize(23961) == 'Ukraine')
        judge_expression(self.tokenizer.detokenize(self.tokenizer.eos) == '</s>')
    
    def test_build_splitter(self):
        """
        If there's no split_sentence, default process is `IdentitySplitter()`.
        """
        pass

    def test_build_dataset(self):
        """
        Test the raw_dataset, need to test number of columns and rows
        """
        judge_expression(len(self.raw_dataset.__getitem__("instruction")) == 52002)
        judge_expression(len(self.raw_dataset.__getitem__("input")) == 52002)
        judge_expression(len(self.raw_dataset.__getitem__("output")) == 52002)
        judge_expression(len(self.raw_dataset.__getitem__("text")) == 52002)
    
    def test_get_dataset_handler(self):
        """
        Test if get the right data handler for pretrain
        """
        judge_expression(isinstance(self.handler, GeneralPretrainHandler))
    
    def test_serialize_to_disk(self):
        """
        Test generate pretrain object files and files are not None(MB).
        """
        self.handler.serialize_to_disk()
        folder_path = "/data/pretrain_dataset"
        bin_file = 0
        idx_file = 0
        total_size = 0
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                if file_path.endswith(".bin") and file_name.startswith('alpaca_'):
                    bin_file += 1
                    total_size += os.path.getsize(file_path)
                if file_path.endswith(".idx") and file_name.startswith('alpaca_'):
                    idx_file += 1
                    total_size += os.path.getsize(file_path)
        judge_expression(bin_file == 1)
        judge_expression(idx_file == 1)
        judge_expression(math.isclose(total_size / (1024 * 1024), 13 * 2, abs_tol=1))
