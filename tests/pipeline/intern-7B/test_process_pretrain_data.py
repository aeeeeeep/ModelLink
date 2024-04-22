import unittest
import sys
import os
import glob
from utils import ParamConfig

import modellink

from modellink.tokenizer import build_tokenizer
from modellink.tokenizer.tokenizer import _AutoTokenizer
from modellink.data.data_handler import AlpacaPretrainHandler
from modellink.data.data_handler import build_dataset, get_dataset_handler
from tools.preprocess_data import get_args, build_splitter


class TestProcessPretrainData(unittest.TestCase):
    def setUp(self, config=ParamConfig):
        sys.argv = [sys.argv[0]] + config.process_pretrain_data
        self.config = config.process_pretrain_data
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
        self.assertIsInstance(self.tokenizer, _AutoTokenizer)
        self.assertEqual(self.tokenizer.vocab_size, 103168)
        self.assertEqual(self.tokenizer.tokenize('bug'), [1, 2463])
        self.assertEqual(self.tokenizer.detokenize(23961), ' possibilities')
        self.assertEqual(self.tokenizer.detokenize(self.tokenizer.eos), '</s>')
    
    def test_build_splitter(self):
        """
        If there's no split_sentence, default process is `IdentitySplitter()`.
        """
        pass

    def test_build_dataset(self):
        """
        Test the raw_dataset, need to test number of columns and rows
        """
        self.assertEqual(len(self.raw_dataset.__getitem__("instruction")), 52002)
        self.assertEqual(len(self.raw_dataset.__getitem__("input")), 52002)
        self.assertEqual(len(self.raw_dataset.__getitem__("output")), 52002)
        self.assertEqual(len(self.raw_dataset.__getitem__("text")), 52002)
    
    def test_get_dataset_handler(self):
        """
        Test if get the right data handler for pretrain
        """
        self.assertIsInstance(self.handler, AlpacaPretrainHandler)
    
    def test_serialize_to_disk(self):
        """
        Test generate pretrain object files and files are not None(MB).
        """
        self.handler.serialize_to_disk()
        folder_path = self.config[5].replace("/alpaca", "")
        bin_file = glob.glob(os.path.join(folder_path, "*.bin"))
        idx_file = glob.glob(os.path.join(folder_path, "*.idx"))
        total_size = 0
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
        self.assertEqual(len(bin_file), 1)
        self.assertEqual(len(idx_file), 1)
        self.assertAlmostEqual((total_size / (1024 * 1024)), 28, delta=1)


if __name__ == "__main__":
    unittest.main()
