import sys
import os
import nltk
import torch
import torch_npu
from ascendspeed import megatron_adaptor
from common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink

from megatron.model import GPTModel
from tasks.inference.infer_base import add_text_generate_args


class TestGeneration(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig):
        """
        initialize the environment and arguments
        """
        sys.argv = [sys.argv[0]] + config.distributed_param + config.network_size + \
                   config.inference_param + config.inference_aux + config.auxiliary_param
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
        from tasks.inference.inference_llama import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )
        instruction = ["春夏秋冬，四个季节"]
        output = model.generate(instruction, detokenize=False)
        
        expected_output = [98899, 67921, 70257, 67780, 60443, 67942, 68212, 98899, 60357, 60443,
        67942, 60515, 98899, 60357, 60443, 67942, 68123, 99157, 364, 61145,
        98899, 60355, 67546, 60353, 62513, 60410, 98899, 60355, 72801, 61209,
        60431, 98899, 60355, 60758, 70447, 83396, 98899, 60355, 60758, 60958,
        60353, 68124, 99157, 364, 61145, 60353, 62513, 60410, 98899, 60355,
        67546, 60353, 62513, 60410, 98899, 60355, 72801, 61209, 60431, 98899,
        ]

        expected_output_seq = torch.tensor(expected_output)[:20].unsqueeze(0).float().npu()
        output_seq = output[:20].unsqueeze(0).float()
    
        if torch.distributed.get_rank() == 0:
            print(len(output))
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(expected_output_seq, output_seq)
            print("cos_sim:", cos_sim)
            assert_judge(cos_sim > 0.80)
    
    def test_beam_search(self):
        """
        load weight to get model and construct the prompts to generate output, 
        and compare with expected for `beam search`.
        """
        self.init(config=ParamConfig)
        from tasks.inference.inference_llama import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )
        max_new_tokens = self.args.max_new_tokens
        instruction = "春夏秋冬，四个季节"
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
        expected_output = [98899, 67921, 70257, 67780, 60724, 71526, 68881, 99157, 60450, 67921,
        70257, 60417, 98899, 60661, 67780, 60724, 60434, 68108, 60477, 61472,
        60353, 76934, 99157, 364, 72196, 98899, 75427, 83396, 99157, 364,
        69025, 98899, 83649, 61549, 60511, 99157, 364, 75814, 98899, 62084,
        60449, 61469, 61469, 99157, 364, 69713, 98899, 61139, 60620, 60862,
        ]
        expected_output_seq = torch.tensor(expected_output)[:15].unsqueeze(0).float().npu()
        output_seq = output[:15].unsqueeze(0).float()
        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(expected_output_seq, output_seq)
            print("cos_sim:", cos_sim)
            assert_judge(cos_sim > 0.6)

