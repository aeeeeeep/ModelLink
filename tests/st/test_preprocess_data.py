# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
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
"""Tests of preprocess data"""

import sys
from pathlib import Path
import itertools
import pytest
from tools.preprocess_data import main
from tests.st.utils import build_args
from tests.common import create_testconfig


class TestPreprocessData:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.fixture
    def outdir(self, tmp_path, request):
        sys.argv.append('--output-prefix')
        prefix = request.getfixturevalue('prefix')
        sys.argv.append(f'{tmp_path}/{prefix}')
        yield tmp_path
    
    @pytest.mark.parametrize("params, prefix", test_config["test_preprocess_pretrain_data"])
    def test_preprocess_pretrain_data(self, build_args, outdir, params, prefix):
        main()
        assert len(list(outdir.glob(f'{prefix}_text_document.bin'))) == 1
        assert len(list(outdir.glob(f'{prefix}_text_document.idx'))) == 1

    @pytest.mark.parametrize("params, prefix", test_config["test_preprocess_instruction_data"])
    def test_preprocess_instruction_data(self, build_args, outdir, params, prefix):
        main()
        for column, suffix in itertools.product(("attention_mask", "input_ids", "labels"), ("bin", "idx")):
            assert len(list(outdir.glob(f'{prefix}_packed_{column}_document.{suffix}'))) == 1
