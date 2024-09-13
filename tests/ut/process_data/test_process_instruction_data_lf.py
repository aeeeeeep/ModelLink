import sys
import os
import math

import pandas as pd

import modellink
from tests.test_tools.utils import judge_expression, get_md5sum
from modellink.tokenizer import build_tokenizer
from modellink.tokenizer.tokenizer import _AutoTokenizer
from modellink.tasks.preprocess.data_handler import AlpacaStyleInstructionHandler, SharegptStyleInstructionHandler
from modellink.tasks.preprocess.data_handler import build_dataset, get_dataset_handler
from preprocess_data import get_args, build_splitter


class TestProcessInstructionDataLf:

    def setup_class(self):
        # test for alpaca
        sys.argv = [
            sys.argv[0],
            "--input", "/data/tune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
            "--tokenizer-type", "PretrainedFromHF",
            "--handler-name", "AlpacaStyleInstructionHandler",
            "--output-prefix", "/data/tune_dataset/alpaca/alpaca",
            "--tokenizer-name-or-path", "/data/qwen-7b/",
            "--workers", "4",
            "--log-interval", "1000",
            "--prompt-type", "qwen"
        ]
        self.args = get_args()
        self.tokenizer = build_tokenizer(self.args)
        self.splitter = build_splitter(self.args)
        self.raw_dataset_alpaca = build_dataset(self.args)
        self.handler_alpaca = get_dataset_handler(self.args, self.raw_dataset_alpaca, self.tokenizer, self.splitter)

        # test for alpaca history
        sys.argv = [
            sys.argv[0],
            "--input", "/data/tune_dataset/oaast_sft.json",
            "--tokenizer-type", "PretrainedFromHF",
            "--handler-name", "AlpacaStyleInstructionHandler",
            "--output-prefix", "/data/tune_dataset/alpaca_his/alpaca_his",
            "--tokenizer-name-or-path", "/data/qwen-7b/",
            "--workers", "4",
            "--log-interval", "1000",
            "--prompt-type", "qwen",
            "--map-keys", '{"history":"history"}'
        ]
        self.args = get_args()
        self.raw_dataset_alpaca_his = build_dataset(self.args)
        self.handler_alpaca_his = get_dataset_handler(self.args, self.raw_dataset_alpaca_his, self.tokenizer, self.splitter)

        # test for sharegpt
        sys.argv = [
            sys.argv[0],
            "--input", "/data/tune_dataset/sharegpt_formatted_data-evol-gpt4.jsonl",
            "--tokenizer-type", "PretrainedFromHF",
            "--handler-name", "SharegptStyleInstructionHandler",
            "--output-prefix", "/data/tune_dataset/sharegpt/sharegpt",
            "--tokenizer-name-or-path", "/data/qwen-7b/",
            "--workers", "4",
            "--log-interval", "1000",
            "--prompt-type", "qwen",
            "--map-keys", '{"system":"system_prompt"}'
        ]

        self.args = get_args()
        self.raw_dataset_sharegpt = build_dataset(self.args)
        self.handler_sharegpt = get_dataset_handler(self.args, self.raw_dataset_sharegpt, self.tokenizer, self.splitter)

        # test for openai
        sys.argv = [
            sys.argv[0],
            "--input", "/data/tune_dataset/sss.json",
            "--tokenizer-type", "PretrainedFromHF",
            "--handler-name", "SharegptStyleInstructionHandler",
            "--output-prefix", "/data/tune_dataset/openai/openai",
            "--tokenizer-name-or-path", "/data/qwen-7b/",
            "--workers", "4",
            "--log-interval", "1000",
            "--prompt-type", "qwen",
            "--map-keys", '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"} }'
        ]

        self.args = get_args()
        self.raw_dataset_openai = build_dataset(self.args)
        self.handler_openai = get_dataset_handler(self.args, self.raw_dataset_openai, self.tokenizer, self.splitter)

    
    def test_get_dataset_handler(self):
        """
        Test if get the right data handler for pretrain
        """
        judge_expression(isinstance(self.handler_alpaca, AlpacaStyleInstructionHandler))
        judge_expression(isinstance(self.handler_alpaca_his, AlpacaStyleInstructionHandler))
        judge_expression(isinstance(self.handler_sharegpt, SharegptStyleInstructionHandler))
        judge_expression(isinstance(self.handler_openai, SharegptStyleInstructionHandler))


    def test_serialize_to_disk(self):
        """
        Test generate pretrain object files and files are not None(MB).
        """
        self.handler_alpaca.serialize_to_disk()
        self.handler_alpaca_his.serialize_to_disk()
        self.handler_sharegpt.serialize_to_disk()
        self.handler_openai.serialize_to_disk()
        folder_path1 = "/data/tune_dataset/alpaca/"
        folder_path2 = "/data/tune_dataset/alpaca_his/"
        folder_path3 = "/data/tune_dataset/sharegpt/"
        folder_path4 = "/data/tune_dataset/openai/"

        def check_file_num(folder_path):
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

        check_file_num(folder_path1)
        check_file_num(folder_path2)
        check_file_num(folder_path3)
        check_file_num(folder_path4)


    def test_md5sum_with_llamafactoryhandler(self):
        file_path_alpaca = "/data/tune_dataset/alpaca/alpaca"
        file_path_alpaca_his = "/data/tune_dataset/alpaca_his/alpaca_his"
        file_path_sharegpt = "/data/tune_dataset/sharegpt/sharegpt"
        file_path_openai = "/data/tune_dataset/openai/openai"

        file_path_compare_alpaca = "/data/tune_dataset/Llamafactoryhandler/alpaca/alpaca"
        file_path_compare_alpaca_his = "/data/tune_dataset/Llamafactoryhandler/alpaca_history/alpaca_history"
        file_path_compare_sharegpt = "/data/tune_dataset/Llamafactoryhandler/sharegpt/sharegpt_lf"
        file_path_compare_openai = "/data/tune_dataset/Llamafactoryhandler/openai/sss"

        def compare_md5sum(file_path1, file_path2):
            judge_expression(get_md5sum(file_path1 + "_packed_attention_mask_document.idx") == get_md5sum(file_path2 + "_packed_attention_mask_document.idx"))
            judge_expression(get_md5sum(file_path1 + "_packed_attention_mask_document.bin") == get_md5sum(file_path2 + "_packed_attention_mask_document.bin"))
            judge_expression(get_md5sum(file_path1 + "_packed_input_ids_document.idx") == get_md5sum(file_path2 + "_packed_input_ids_document.idx"))
            judge_expression(get_md5sum(file_path1 + "_packed_input_ids_document.bin") == get_md5sum(file_path2 + "_packed_input_ids_document.bin"))
            judge_expression(get_md5sum(file_path1 + "_packed_labels_document.idx") == get_md5sum(file_path2 + "_packed_labels_document.idx"))
            judge_expression(get_md5sum(file_path1 + "_packed_labels_document.bin") == get_md5sum(file_path2 + "_packed_labels_document.bin"))

        compare_md5sum(file_path_alpaca, file_path_compare_alpaca)
        compare_md5sum(file_path_alpaca_his, file_path_compare_alpaca_his)
        compare_md5sum(file_path_sharegpt, file_path_compare_sharegpt)
        compare_md5sum(file_path_openai, file_path_compare_openai)

