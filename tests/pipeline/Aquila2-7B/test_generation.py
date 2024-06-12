import sys
import os
import torch
import torch_npu
from utils import ParamConfig, assert_judge
import modellink
from megatron.legacy.model import GPTModel
from modellink.tasks.inference.text_generation.infer_base import add_text_generate_args
from tests.pipeline.common import DistributedTest


class TestGeneration(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig):
        """
        initialize the environment and arguments
        """
        sys.argv = [sys.argv[0]] + config.distributed_param + config.network_size + \
                   config.inference_param + config.auxiliary_param + config.tokenizer_param
        from megatron.training.initialize import initialize_megatron
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
        from megatron.training import get_args
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
        instruction = ["解释一下“温故而知新”"]
        output = model.generate(instruction, detokenize=False)
        expected_output = [14727, 29728, 261, 22051, 1156, 3101, 422, 278, 0, 87,
                           18392, 221, 1906, 358, 132, 237, 79, 221, 261, 81,
                           14572, 2449, 2369, 72, 8022, 2449, 221, 261, 88, 14572]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output)[:20].unsqueeze(0).float().npu(),
                                 output[:20].unsqueeze(0).float())
            print(cos_sim)
            assert_judge(cos_sim > 0.80)

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
        prompt = "解释一下“温故而知新”"
        system_template = ""
        dialog_template = "{instruction}"
        template = system_template + dialog_template
        instruction = template.format(instruction=prompt)

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
        expected_output = [391, 426, 1774, 1906, 8627, 10281, 261, 36018, 837, 42310,
                           434, 10045, 6468, 29728, 278, 0, 43652, 43652, 1774, 1906,
                           28043, 609, 43652, 43652, 1774, 1906, 8627, 10281, 261, 36018]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output)[:20].unsqueeze(0).float().npu(),
                                 output[:20].unsqueeze(0).float())
            print(cos_sim)
            assert_judge(cos_sim > 0.80)
