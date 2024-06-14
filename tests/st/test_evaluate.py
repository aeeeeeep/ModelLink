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
"""Tests of Evaluation"""

import sys
from pathlib import Path
import pytest
from evaluation import main
from tests.common import DistributedTest, create_testconfig
from tests.st.utils import build_args


class TestEvaluate(DistributedTest):
    world_size = 4
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))


    @pytest.fixture
    def env_setup(self, monkeypatch):
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    
    @pytest.mark.parametrize("params", test_config["test_evaluate"])
    def test_evaluate(self, build_args, env_setup, params):
        main()
