import os
import pytest
from pathlib import Path
import pandas as pd
import modellink
from test_tools.utils import build_args, create_testconfig, setup_logger, compare_files_md5
from preprocess_data import main as preprocess_datasets_main
from merge_datasets import main as merge_datasets_main


PATTERN = "preprocess"


class TestProcessPretrainData:
    """
        The unified dataset is divided into two parts, 
        individual processing results as well as results from the combined dataset.
        The three designed test cases are as follows: 
        1. processing of the first segment of the split dataset
        2. processing of the second segment of the split dataset
        3. merging the two segments and processing them together.
    """

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    @pytest.mark.parametrize("full_params, params",
        [(test_config["process_dataset"][0], test_config["test_preprocess_datasets_part1"][0])])
    def test_preprocess_datasets_part1(self, build_args, full_params, params):
        if not os.path.isdir(full_params["test-out-part"]):
            os.makedirs(full_params["test-out-part"])
        df = pd.read_parquet(full_params["input-dataset"])
        df.iloc[:25000, :].to_parquet(params["input"])
        
        handler, log_capture = setup_logger(PATTERN)
        preprocess_datasets_main()
        print("=============== preprocess datasets part1 =============")
        print(log_capture)

        prefix_str = params["output-prefix"].split('/')[-1]
        end_strs = ["_text_document.bin", "_text_document.idx"]
        for end_str in end_strs:
            test_file = params["output-prefix"] + end_str
            end_str = prefix_str + end_str
            base_file = full_params["base-out-part"] + end_str
            assert compare_files_md5(base_file, test_file)


    @pytest.mark.parametrize("full_params, params",
        [(test_config["process_dataset"][0], test_config["test_preprocess_datasets_part2"][0])])
    def test_preprocess_datasets_part2(self, build_args, full_params, params):
        if not os.path.isdir(full_params["test-out-part"]):
            os.makedirs(full_params["test-out-part"])
        df = pd.read_parquet(full_params["input-dataset"])
        df.iloc[25000:, :].to_parquet(params["input"])
        
        handler, log_capture = setup_logger(PATTERN)
        preprocess_datasets_main()
        print("=============== preprocess datasets part2 =============")
        print(log_capture)

        prefix_str = params["output-prefix"].split('/')[-1]
        end_strs = ["_text_document.bin", "_text_document.idx"]
        for end_str in end_strs:
            test_file = params["output-prefix"] + end_str
            end_str = prefix_str + end_str
            base_file = full_params["base-out-part"] + end_str
            assert compare_files_md5(base_file, test_file)
    

    @pytest.mark.parametrize("full_params, params",
        [(test_config["process_dataset"][0], test_config["test_merge_datasets"][0])])
    def test_merge_datasets(self, build_args, full_params, params):
        if not os.path.isdir(full_params["test-out-merge"]):
            os.makedirs(full_params["test-out-merge"])

        handler, log_capture = setup_logger(PATTERN)
        merge_datasets_main()
        print("=============== merge pretrain datasets =============")
        print(log_capture)

        prefix_str = params["output-prefix"].split('/')[-1]
        end_strs = ["_text_document.bin", "_text_document.idx"]
        for end_str in end_strs:
            test_file = params["output-prefix"] + end_str
            end_str = prefix_str + end_str
            base_file = full_params["base-out-merge"] + end_str
            assert compare_files_md5(base_file, test_file)
