#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

import sys
import os
import torch
import torch_npu
from common import DistributedTest
from utils import ParamConfig, assert_judge
from megatron.model import GPTModel

import modellink
from modellink.tasks.inference.text_generation.infer_base import add_text_generate_args


class TestGeneration(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig):
        """
        initialize the environment and arguments
        """
        sys.argv = [sys.argv[0]] + config.distributed_param + config.network_size + \
                   config.inference_param + config.auxiliary_param + config.tokenizer_param
        from megatron.initialize import initialize_megatron
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
        from megatron import get_args
        self.args = get_args()

    def test_greedy_search(self):
        """
        load weight to get model and construct the prompts to generate output,
        and compare with expected for `greedy search`.
        """
        self.init(config=ParamConfig)
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )
        instruction = ["春夏秋冬，四个季节"]
        output = model.generate(instruction, detokenize=False)
        expected_output1 = [15946, 3837, 48309, 99369, 77959, 9370, 106447, 100682, 20412, 9909,
                            22441, 22441, 23083, 32, 13, 38903, 98, 99377, 106447, 103994,
                            198, 33, 13, 40666, 237, 99377, 106447, 103994, 198, 34,
                            13, 75671, 233, 99377, 106447, 103994, 198, 35, 13, 68739,
                            105, 99377, 106447, 103994, 198, 102349, 5122, 33, 198, 49238]

        expected_output2 = [15946, 3837, 48309, 99369, 77959, 9370, 109137, 20412, 9909, 22441,
                            22441, 23083, 32, 13, 220, 18, 5373, 19, 5373, 20,
                            108213, 198, 33, 13, 220, 21, 5373, 22, 5373, 23,
                            108213, 198, 34, 13, 220, 24, 5373, 16, 15, 5373,
                            16, 16, 108213, 198, 35, 13, 220, 16, 17, 5373]

        expected_output3 = [15946, 3837, 48309, 99369, 77959, 9370, 109137, 20412, 9909, 22441,
                            22441, 23083, 32, 13, 220, 18, 5373, 19, 5373, 20,
                            9754, 198, 33, 13, 220, 24, 5373, 16, 15, 5373,
                            16, 16, 9754, 198, 34, 13, 220, 16, 15, 5373,
                            16, 16, 5373, 16, 17, 9754, 198, 35, 13, 220]

        expected_output4 = [3837, 56568, 112795, 104673, 105419, 11319, 100678, 11319, 61443, 14777,
                            61443, 100003, 6313, 198, 2130, 198, 109089, 28311, 35946, 112795,
                            105303, 198, 105303, 101161, 3837, 108441, 108105, 96050, 43288, 106369,
                            105129, 69249, 3837, 35946, 112795, 85336, 102077, 99366, 8997, 104400,
                            102077, 3837, 101140, 100261, 17254, 99246, 103287, 105729, 100230, 104679]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = torch.tensor(0.0).to(device=output.device)
            for expected_output in [expected_output1, expected_output2, expected_output3, expected_output4]:
                new_cos_sim = similarity(torch.tensor(expected_output).unsqueeze(0).float().npu(),
                                         output[:50].unsqueeze(0).float())
                cos_sim = torch.max(cos_sim, new_cos_sim)
            print("similarity: ", cos_sim)
            assert_judge(cos_sim > 0.8)

    def test_beam_search(self):
        """
        load weight to get model and construct the prompts to generate output,
        and compare with expected for `beam search`.
        """
        self.init(config=ParamConfig)
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )

        max_new_tokens = self.args.max_new_tokens
        instruction = "北京奥运会"
        output = model.generate(
            instruction,
            num_beams=2,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            max_new_tokens=max_new_tokens,
            tokenizer=None,
            stream=False,
            detokenize=False
        )
        expected_output1 = [116097, 2073, 102197, 99718, 97907, 70500, 100418, 107614, 99891, 57218,
                            100377, 9370, 101164, 1773, 116097, 109262, 17714, 104165, 3837, 106411,
                            99370, 99755, 5373, 113662, 33108, 100372, 19108, 1773, 116097, 11622,
                            9370, 103159, 17714, 106980, 116356, 9909, 34, 18, 39, 23,
                            48272, 41146, 106578, 107795, 17714, 114804, 33108, 52510, 1773, 107976]

        expected_output2 = [116097, 2073, 102197, 99718, 97907, 70500, 100418, 107614, 99891, 57218,
                            100377, 9370, 101164, 1773, 116097, 109262, 103963, 108361, 3837, 41146,
                            108304, 9370, 106578, 72448, 31843, 98641, 101117, 24300, 103159, 106980,
                            116356, 9909, 34, 18, 39, 23, 48272, 106578, 33447, 42192,
                            100791, 3837, 107614, 2073, 100706, 103010, 97907, 100427, 1773, 107976]

        expected_output3 = [109076, 17447, 70074, 99272, 102276, 99750, 2073, 52726, 119613, 68536,
                            99678, 854, 100848, 99631, 100794, 3837, 119613, 70074, 20412, 101887,
                            119613, 27091, 6567, 234, 107, 27733, 220, 102710, 3837, 101889,
                            67338, 10236, 102, 118, 99180, 220, 101170, 26939, 100647, 104143,
                            107704, 9370, 8997, 49238, 5122, 119613, 70074, 20412, 101887, 119613]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = torch.tensor(0.0).to(device=output.device)
            for expected_output in [expected_output1, expected_output2, expected_output3]:
                new_cos_sim = similarity(torch.tensor(expected_output).unsqueeze(0).float().npu(),
                                         output[:50].unsqueeze(0).float())
                cos_sim = torch.max(cos_sim, new_cos_sim)
            print("similarity: ", cos_sim)
            assert_judge(cos_sim > 0.8)
