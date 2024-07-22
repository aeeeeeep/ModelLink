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

import unittest
import sys
import os
import glob
from utils import ParamConfig

import modellink

from modellink.tokenizer import build_tokenizer
from modellink.tokenizer.tokenizer import _AutoTokenizer
from modellink.tasks.preprocess.data_handler import LlamaFactoryInstructionHandler
from modellink.tasks.preprocess.data_handler import build_dataset, get_dataset_handler
from tools.preprocess_data import get_args, build_splitter


class TestProcessInstructionData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # configure params, the index starts from 1
        self.config = ParamConfig
        sys.argv = [sys.argv[0]] + self.config.instruction_data_param
        self.args = get_args()
        self.tokenizer = build_tokenizer(self.args)
        self.splitter = build_splitter(self.args)
        self.raw_dataset = build_dataset(self.args)
        self.handler = get_dataset_handler(self.args, self.raw_dataset, self.tokenizer, self.splitter)

        # for test_build_dataset_mix1
        sys.argv = [sys.argv[0]] + self.config.instruction_data_mix_param1
        self.args = get_args()
        self.raw_dataset_mix1 = build_dataset(self.args)

        # for test_build_dataset_mix2
        sys.argv = [sys.argv[0]] + self.config.instruction_data_mix_param2
        self.args = get_args()
        self.raw_dataset_mix2 = build_dataset(self.args)


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
        self.assertIsInstance(self.tokenizer, _AutoTokenizer)

        self.assertEqual(self.tokenizer.vocab_size, 151851)
        self.assertEqual(self.tokenizer.tokenize('<0xF7>'), [27, 15, 9770, 22, 29])
        self.assertEqual(self.tokenizer.detokenize(31338), '�建')
        self.assertEqual(self.tokenizer.detokenize(self.tokenizer.eod), '<|im_end|>')


    def test_build_splitter(self):
        """
        If there's no split_sentence, default process is `IdentitySplitter()`.
        """
        pass


    def test_build_dataset_mix1(self):
        """
        Test the raw_dataset, need to test number of columns and rows
            outputs["prompt"] = prompt
            outputs["response"] = response
            outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
            outputs["tools"].append("")
        """
        print("-------------------test_build_dataset_mix1-------------------------")
        print(len(self.raw_dataset_mix1.__getitem__("prompt")))
        print(len(self.raw_dataset_mix1.__getitem__("response")))
        print(len(self.raw_dataset_mix1.__getitem__("system")))
        print(len(self.raw_dataset_mix1.__getitem__("tools")))

        self.assertLessEqual(len(self.raw_dataset_mix1.__getitem__("prompt")), 40)
        self.assertLessEqual(len(self.raw_dataset_mix1.__getitem__("response")), 40)
        self.assertLessEqual(len(self.raw_dataset_mix1.__getitem__("system")), 40)
        self.assertLessEqual(len(self.raw_dataset_mix1.__getitem__("tools")), 40)


    def test_build_dataset_mix2(self):
        """
        Test the raw_dataset, need to test number of columns and rows
            outputs["prompt"] = prompt
            outputs["response"] = response
            outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
            outputs["tools"].append("")
        """
        print("----------------test_build_dataset_mix2--------------------------")
        print(len(self.raw_dataset_mix2.__getitem__("prompt")))
        print(len(self.raw_dataset_mix2.__getitem__("response")))
        print(len(self.raw_dataset_mix2.__getitem__("system")))
        print(len(self.raw_dataset_mix2.__getitem__("tools")))

        self.assertGreaterEqual(len(self.raw_dataset_mix2.__getitem__("prompt")), 40)
        self.assertGreaterEqual(len(self.raw_dataset_mix2.__getitem__("response")), 40)
        self.assertGreaterEqual(len(self.raw_dataset_mix2.__getitem__("system")), 40)
        self.assertGreaterEqual(len(self.raw_dataset_mix2.__getitem__("tools")), 40)


    def test_build_dataset(self):
        """
        Test the raw_dataset, need to test number of columns and rows
            outputs["prompt"] = prompt
            outputs["response"] = response
            outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
            outputs["tools"].append("")
        """
        self.assertEqual(len(self.raw_dataset.__getitem__("prompt")), 62981)
        self.assertEqual(len(self.raw_dataset.__getitem__("response")), 62981)
        self.assertEqual(len(self.raw_dataset.__getitem__("system")), 62981)
        self.assertEqual(len(self.raw_dataset.__getitem__("tools")), 62981)


    def test_get_dataset_handler(self):
        """
        Test if get the right data handler for pretrain
        """
        self.assertIsInstance(self.handler, LlamaFactoryInstructionHandler)


    def test_serialize_to_disk(self):
        """
        Test generate pretrain object files and files are not None(MB).
        """
        self.handler.serialize_to_disk()
        folder_path = self.config.instruction_data_param[7].replace("/alpaca", "")
        bin_file = glob.glob(os.path.join(folder_path, "*.bin"))
        idx_file = glob.glob(os.path.join(folder_path, "*.idx"))
        total_size = 0
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
        self.assertEqual(len(bin_file), 3)
        self.assertEqual(len(idx_file), 3)
        self.assertAlmostEqual((total_size / (1024 * 1024)), 111, delta=1)


if __name__ == "__main__":
    unittest.main()