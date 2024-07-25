import sys
import os
import json
from pathlib import Path
import tqdm
import pandas as pd
import torch
import torch_npu
from transformers import AutoTokenizer
from tests.common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink
from megatron.legacy.model import GPTModel
from modellink.tasks.evaluation.utils import add_text_generate_args
from evaluation import mmlu, boolq, ceval, LLMChat


class TestEvaluation(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig, task=None):

        sys.argv = [sys.argv[0]] + config.distributed_param_tp8_pp1 + config.network_size + config.auxiliary_param + \
                   config.inference_param + config.tokenizer_param
        if task == "mmlu":
            sys.argv = sys.argv + config.evaluation_param_mmlu
        elif task == "boolq":
            sys.argv = sys.argv + config.evaluation_param_boolq
        elif task == "ceval":
            sys.argv = sys.argv + config.evaluation_param_ceval

        from megatron.training.initialize import initialize_megatron
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})

        from megatron.training import get_args
        self.args = get_args()


    def test_mmlu_evaluation(self):
        self.init(config=ParamConfig, task="mmlu")
        from evaluation import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_name_or_path=self.args.load
        )
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, trust_remote_code=True)
        _, score_df = mmlu(self.args, LLMChat(self.args, model, tokenizer))
        if torch.distributed.get_rank() == 0:
            print("-------------score---------------")
            print(score_df.iloc[-1][-1])
            assert_judge(score_df.iloc[-1][-1] >= 0.50)


    def test_boolq_evaluation(self):
        self.init(config=ParamConfig, task="boolq")
        from evaluation import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_name_or_path=self.args.load
        )
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, trust_remote_code=True)
        _, score_df = boolq(self.args, LLMChat(self.args, model, tokenizer))
        if torch.distributed.get_rank() == 0:
            print("-------------score---------------")
            print(score_df.iloc[-1][-1])
            assert_judge(score_df.iloc[-1][-1] >= 0.50)


    def test_ceval_evaluation(self):
        self.init(config=ParamConfig, task="ceval")
        from evaluation import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_name_or_path=self.args.load
        )
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, trust_remote_code=True)
        _, score_df = ceval(self.args, LLMChat(self.args, model, tokenizer))
        if torch.distributed.get_rank() == 0:
            print("-------------score---------------")
            print(score_df.iloc[-1][-1])
            assert_judge(score_df.iloc[-1][-1] >= 0.5)