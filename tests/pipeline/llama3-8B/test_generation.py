import sys
import os
import torch
import torch_npu
from tests.common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink
from megatron.legacy.model import GPTModel
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from modellink.tasks.inference.text_generation.infer_base import add_text_generate_args


class TestGeneration(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig):
        """
        initialize the environment and arguments
        """
        sys.argv = [sys.argv[0]] + config.distributed_param + config.network_size + \
                   config.inference_param + config.auxiliary_param + config.tokenizer_param
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
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
        expected_output1 = [3922, 64803, 19483, 105343, 56602, 3922, 64803, 19483, 105343, 56602,
                            3922, 64803, 19483, 105343, 56602, 3922, 64803, 19483, 105343, 56602,
                            3922, 64803, 19483, 105343, 56602, 3922, 64803, 19483, 105343, 56602,
                            3922, 64803, 19483, 105343, 56602, 3922, 64803, 19483, 105343, 56602]
        expected_output2 = [3922, 64803, 19483, 13646, 125436, 3922, 64803, 19483, 24273, 25129,
                            3922, 64803, 19483, 27384, 24273, 25129, 3922, 64803, 19483, 31809,
                            24273, 25129, 3922, 64803, 19483, 27384, 24273, 25129, 9554, 64803,
                            19483, 31809, 24273, 25129, 3922, 64803, 19483, 31809, 24273, 25129]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim1 = similarity(torch.tensor(expected_output1).unsqueeze(0).float().npu(),
                                 output[:40].unsqueeze(0).float())
            cos_sim2 = similarity(torch.tensor(expected_output2).unsqueeze(0).float().npu(),
                                 output[:40].unsqueeze(0).float())
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
        expected_output = [9554, 30867, 106633, 29430, 17905, 3922, 102446, 110125, 35287, 28038,
                           70090, 108025, 109169, 57668, 26123, 34208, 28038, 37046, 34208, 57668,
                           26123, 78640, 61075, 104261, 103302, 1811, 1049, 23, 8107, 24,
                           9953, 3922, 110284, 35287, 19000, 70090, 108448, 23039, 9554, 30537]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output).unsqueeze(0).float().npu(),
                                 output[:40].unsqueeze(0).float())
            print("similarity: ", cos_sim)
            assert_judge(cos_sim > 0.95)
