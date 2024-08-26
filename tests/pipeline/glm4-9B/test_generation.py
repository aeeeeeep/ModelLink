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
    world_size = 4

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
        expected_output1 = [3837, 100455, 103787, 100143, 100370, 100217, 3837, 100455, 103787, 100143,
                            100370, 98707, 3837, 100455, 103787, 100143, 100370, 103444, 3837, 100455,
                            103787, 100143, 100370, 102851, 3837, 100455, 103787, 100143, 100370, 100787,
                            3837, 100455, 103787, 100143, 100370, 110565, 3837, 100455, 103787, 100143]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output1).unsqueeze(0).float().npu(),
                                 output[:40].unsqueeze(0).float())
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
        expected_output = [113618, 98332, 3837, 99435, 103320, 98319, 104550, 98923, 101640, 107295,
                           3837, 100572, 98369, 98668, 98328, 101640, 99702, 3837, 99245, 105631,
                           3837, 99435, 103320, 98319, 104550, 98923, 101640, 107295, 100572, 102176,
                           98668, 98328, 101640, 99702, 3837, 110316, 98624, 98347, 98359, 100572]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output).unsqueeze(0).float().npu(),
                                 output[:40].unsqueeze(0).float())
            print("similarity: ", cos_sim)
            assert_judge(cos_sim > 0.95)
