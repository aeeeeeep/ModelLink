import sys
import os
import torch
import torch_npu
from common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink
from megatron.model import GPTModel
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
        instruction = ["import socket\n\ndef ping_exponential_backoff(host: str):"]
        output = model.generate(instruction, detokenize=False)
        expected_output1 = [13, 1678, 9995, 13, 1678, 349, 292, 263, 3495, 773,
                            25658, 1250, 2696, 29889, 13, 1678, 9995, 13, 1678, 363,
                            474, 297, 3464, 29898, 29896, 29892, 29871, 29896, 29900, 1125,
                            13, 4706, 1018, 29901, 13, 9651, 9909, 29889, 29887, 621,
                            520, 29890, 948, 420, 29898, 3069, 29897, 13, 9651, 736]
        expected_output2 = [13, 1678, 9995, 13, 1678, 349, 292, 263, 3495, 773,
                            25658, 1250, 2696, 29889, 13, 1678, 9995, 13, 1678, 9055,
                            353, 29871, 29896, 13, 1678, 1550, 5852, 29901, 13, 4706,
                            1018, 29901, 13, 9651, 9909, 29889, 29887, 621, 520, 29890,
                            948, 420, 29898, 3069, 29897, 13, 9651, 736, 13, 4706]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim1 = similarity(torch.tensor(expected_output1).unsqueeze(0).float().npu(),
                                 output[:50].unsqueeze(0).float())
            cos_sim2 = similarity(torch.tensor(expected_output2).unsqueeze(0).float().npu(),
                                 output[:50].unsqueeze(0).float())
            cos_sim = torch.max(cos_sim1, cos_sim2)
            print("similarity: ", cos_sim)
            assert_judge(cos_sim > 0.95)

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
        instruction = "def fibonacci("
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
        expected_output = [29876, 1125, 13, 1678, 565, 302, 1275, 29871, 29900, 29901,
                           13, 4706, 736, 29871, 29900, 13, 1678, 25342, 302, 1275,
                           29871, 29896, 29901, 13, 4706, 736, 29871, 29896, 13, 1678,
                           1683, 29901, 13, 4706, 736, 18755, 265, 21566, 29898, 29876,
                           448, 29871, 29896, 29897, 718, 18755, 265, 21566, 29898, 29876]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output).unsqueeze(0).float().npu(),
                                 output[:50].unsqueeze(0).float())
            print("similarity: ", cos_sim)
            assert_judge(cos_sim > 0.95)
