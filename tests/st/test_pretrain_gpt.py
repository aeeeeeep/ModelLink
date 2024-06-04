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
"""Tests of pretrain gpt"""

import sys
from pathlib import Path
import pytest
from pretrain_gpt import main
from tests.common import DistributedTest, create_testconfig
from tests.st.utils import build_args


class TestPretrainGPT(DistributedTest):
    world_size = 8
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.fixture
    def outdir(self, tmp_path):
        sys.argv.append('--save')
        sys.argv.append(str(tmp_path))
        yield tmp_path
    
    @pytest.fixture
    def env_setup(self, monkeypatch):
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        # can't use multi num dataloader workers in distributed test
        sys.argv.append('--num-workers')
        sys.argv.append('0')


    @pytest.mark.parametrize("params", test_config["test_pretrain"])
    def test_pretrain(self, build_args, outdir, env_setup, params):
        main()
