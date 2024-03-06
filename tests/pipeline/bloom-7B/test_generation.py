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
        sys.argv = [sys.argv[0]] + config.network_size + config.distributed_param + \
                    config.inference_param + config.inference_aux + \
                    config.auxiliary_param + config.regularization
        from megatron.initialize import initialize_megatron
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
        from megatron import get_args
        self.args = get_args()

    def edit_distance_similarity(self, text1, text2):
        """
        edit distance: to compare the similarity between two texts.
        """
        distance = nltk.edit_distance(text1, text2)
        try:
            similarity = 1 - (distance / max(len(text1), len(text2)))
        except ZeroDivisionError as e:
            raise e
        return similarity

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
        instruction = ["how are you?"]
        output = model.generate(instruction)
        
        expect_output1 = [
            '"\n\n"I\'m all right," ',
            'said the boy, with a smile. ',
            '"I was just thinking of\nyou."',
            '\n\n"Of me?"\n\n"Yes."\n\n"Of course."\n\n"',
            'How are you getting along?"\n\n"',
            'I\'m getting along all right."\n\n"',
            'How are you getting along?"\n\n"'
        ]
        expect_output1_seq = "".join(expect_output1)
        if torch.distributed.get_rank() == 0:
            print(output)
            similarity1 = self.edit_distance_similarity(output[:30], expect_output1_seq[:30])
            print("similarity1:", similarity1)
            assert_judge(similarity1 > 0.9)
   
    
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
        instruction = "What is the whether like today?"
        output = model.generate(
            instruction,
            num_beams=2,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            max_new_tokens=max_new_tokens,
            tokenizer=None,
            stream=False
        )

        expected_output = [
            '"\n\n"It is like a day in the year,"',
            ' said the old man, ',
            '"when the sun shines\nbrightly, and the air is warm,',
            ' and the grass is green, and the flowers\n',
            'bloom, and the birds sing, and the people are happy.',
            '"\n\n"Is it like a day in the year?"',
            ' said the boy.\n\n"',
            'It is like a day in the year,"'
        ]
        expected_output_seq = "".join(expected_output)
        if torch.distributed.get_rank() == 0:
            similarity = self.edit_distance_similarity(output[:20], expected_output_seq[:20])
            print([output])
            print("similarity:", similarity)
            assert_judge(similarity >= 0.6)   
