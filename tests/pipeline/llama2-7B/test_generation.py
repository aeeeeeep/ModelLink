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
                   config.inference_param + config.auxiliary_param
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
        instruction = ["how are you?", "Give me three tips for staying healthy."]
        output = model.generate(instruction)
        expect_output1 = [
            "I'm doing well, thanks for asking! I've been keeping busy with work and spending time with friends and family. ",
            "It's been great to have some time off from school and just relax a bit. How about you? How have you been?\n",
            "\nI hope you're doing well! It's always great to catch up with you and hear about what's going on in your life. ",
            "I'm looking forward to hearing all about it. Let me know if you want to hang out soon!"
        ]
        expect_output2 = [
            '\n\n1. Eat a balanced diet: A healthy diet should include a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats. ',
            'Aim to include a rainbow of colors on your plate to ensure you are getting a range of vitamins and minerals.',
            '\n2. Stay hydrated: Drink plenty of water throughout the day, aiming for at least eight cups (64 ounces) daily. ',
            'Limit your consumption of sugary drinks'
        ]
        
        expect_output1_seq = "".join(expect_output1)
        expect_output2_seq = ''.join(expect_output2)
        
        if torch.distributed.get_rank() == 0:
            print(output[0])
            print(output[1])
            
            similarity1 = self.edit_distance_similarity(output[0][:30], expect_output1_seq[:30])
            similarity2 = self.edit_distance_similarity(output[1][:30], expect_output2_seq[:30])
            print("similarity1:", similarity1)
            print("similarity2:", similarity2)
            assert_judge(similarity1 > 0.85)
            assert_judge(similarity2 > 0.85)
    
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
            "Answer:\nThe weather today is sunny with a high of 75 degrees Fahrenheit and a low of 50 degrees Fahrenheit. ",
            "There is no rain or other weather alerts in the area.",
            "\nWould you like to know the weather for a different location?"
        ]
        expected_output_seq = "".join(expected_output)
        if torch.distributed.get_rank() == 0:
            similarity = self.edit_distance_similarity(output[:40], expected_output_seq[:40])
            print(output)
            print("similarity:", similarity)
            assert_judge(similarity > 0.75)
