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
"""Test of Rotary Pos Embedding"""
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
import mindspeed
import modellink
from tests.common import create_testconfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


class TestRotaryPosEmbedding:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.fixture
    def mock_dependency(self, request, monkeypatch):
        monkeypatch.setattr(modellink.core.models.common.embeddings.rotary_pos_embedding, "get_args",
                            lambda : SimpleNamespace(use_glm_rope=request.getfixturevalue("chatglm")))
        monkeypatch.setattr(mindspeed.core.fusions.rotary_pos_embedding, "get_args",
                            lambda : SimpleNamespace(rotary_base=request.getfixturevalue("rotary_base")))

    @pytest.mark.parametrize("rotary_param, chatglm, rotary_base, seq, expected", test_config["test_rotary_pos_embedding"])

    def test_rotary_pos_embedding(self, mock_dependency, rotary_param, chatglm, rotary_base, seq, expected):
        rotary = RotaryEmbedding(**rotary_param)
        assert(torch.allclose(rotary.forward(seq).cpu(), torch.Tensor(expected)))
        