import sys
import os
import math

from utils import judge_expression

from modellink.tokenizer import build_tokenizer
from modellink.tokenizer.tokenizer import _AutoTokenizer
from modellink.tasks.preprocess.data_handler import GeneralInstructionHandler
from modellink.tasks.preprocess.data_handler import build_dataset, get_dataset_handler
from preprocess_data import get_args, build_splitter


class TestProcessInstructionData:

    def setup_class(self):
        sys.argv = [
            sys.argv[0],
            "--input", "/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
            "--tokenizer-type", "PretrainedFromHF",
            "--handler-name", "GeneralInstructionHandler",
            "--output-prefix", "/data/tune_dataset/alpaca",
            "--tokenizer-name-or-path", "/data/llama-2-7b-hf",
            "--workers", "4",
            "--log-interval", "1000",
            "--append-eod"
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
            the eod append
            ...(If missed something else, welcome to add)
        """
        judge_expression(isinstance(self.tokenizer, _AutoTokenizer))
        judge_expression(self.tokenizer.vocab_size == 32000)
        judge_expression(self.tokenizer.tokenize('<0xF7>') == [1, 529, 29900, 29916, 29943, 29955, 29958])
        judge_expression(self.tokenizer.detokenize(31338) == '堂')
        judge_expression(self.tokenizer.detokenize(self.tokenizer.eod) == '</s>')
    
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
        judge_expression(isinstance(self.handler, GeneralInstructionHandler))
    
    def test_serialize_to_disk(self):
        """
        Test generate pretrain object files and files are not None(MB).
        """
        self.handler.serialize_to_disk()
        folder_path = "/data/tune_dataset"
        bin_file = 0
        idx_file = 0
        total_size = 0
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                if file_path.endswith(".bin"):
                    bin_file += 1
                if file_path.endswith(".idx"):
                    idx_file += 1
                total_size += os.path.getsize(file_path)
        judge_expression(bin_file == 3)
        judge_expression(idx_file == 3)
        judge_expression(math.isclose(total_size / (1024 * 1024), 48 * 2, abs_tol=3))
