import os
import pytest
from pathlib import Path
import pandas as pd
import modellink
from test_tools.utils import build_args, create_testconfig, setup_logger, compare_files_md5
from preprocess_data import main as preprocess_datasets_main
from merge_datasets import main as merge_datasets_main


PATTERN = "preprocess"

class TestProcessInstructionData:
    """
        The unified dataset is divided into two parts, 
        individual processing results as well as results from the combined dataset.
        The three designed test cases are as follows: 
        1. processing of the first segment of the split dataset
        2. processing of the second segment of the split dataset
        3. merging the two segments and processing them together.
    """

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("full_params, params, merge_params", 
        [(test_config["instruction_dataset"][0], test_config["test_datasets_part1"][0], test_config["test_merge_datasets"][0])])
    def test_instruction_datasets_part1(self, build_args, full_params, params, merge_params):
        if not os.path.isdir(full_params["test-out-part"]):
            os.makedirs(full_params["test-out-part"])
        df = pd.read_parquet(full_params["input-dataset"])
        df.iloc[:25000, :].to_parquet(params["input"])
        
        handler, log_capture = setup_logger(PATTERN)
        preprocess_datasets_main()
        print("=============== instruction datasets part1 =============")
        print(log_capture)

        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = [merge_params["keys"][0], merge_params["keys"][1], merge_params["keys"][2]]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-part"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_files_md5(base_file, test_file)
                

    @pytest.mark.parametrize("full_params, params, merge_params", 
        [(test_config["instruction_dataset"][0], test_config["test_datasets_part2"][0], test_config["test_merge_datasets"][0])])
    def test_instruction_datasets_part2(self, build_args, full_params, params, merge_params):
        if not os.path.isdir(full_params["test-out-part"]):
            os.makedirs(full_params["test-out-part"])
        df = pd.read_parquet(full_params["input-dataset"])
        df.iloc[25000:, :].to_parquet(params["input"])
        
        handler, log_capture = setup_logger(PATTERN)
        preprocess_datasets_main()
        print("=============== instruction datasets part2 =============")
        print(log_capture)

        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = [merge_params["keys"][0], merge_params["keys"][1], merge_params["keys"][2]]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-part"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_files_md5(base_file, test_file)

    
    @pytest.mark.parametrize("full_params, params", 
        [(test_config["instruction_dataset"][0], test_config["test_merge_datasets"][0])])
    def test_merge_instruction_datasets(self, build_args, full_params, params):
        if not os.path.isdir(full_params["test-out-merge"]):
            os.makedirs(full_params["test-out-merge"])

        handler, log_capture = setup_logger(PATTERN)
        merge_datasets_main()
        print("=============== merge instruction datasets =============")
        print(log_capture)

        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = [params["keys"][0], params["keys"][1], params["keys"][2]]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-merge"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_files_md5(base_file, test_file)
