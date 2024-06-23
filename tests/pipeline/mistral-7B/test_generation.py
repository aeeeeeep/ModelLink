import sys
import os
import nltk
import torch
import torch_npu
from tests.common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink
from megatron.legacy.model import GPTModel
from modellink.tasks.inference.text_generation.infer_base import add_text_generate_args


class TestGeneration(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig):
        """
        initialize the environment and arguments
        """
        sys.argv = [sys.argv[0]] + config.distributed_param + config.network_size + config.auxiliary_param +\
                   config.inference_aux + config.inference_param
        from megatron.training.initialize import initialize_megatron
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
        from megatron.training import get_args
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
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )
        instruction = ["how are you?", "Give me three tips for staying healthy."]
        output = model.generate(instruction)

        expect_output1 = [
            "I'm doing well, thank you for asking! I've been keeping busy with work and various projects. "
            "How about you? How have you been?"
        ]
        expect_output2 = [
            '\n\n1. Eat a balanced diet: Consuming a variety of nutrient-dense foods from all the food groups is '
            'essential for maintaining good health.\n\n2. Stay hydrated: Water is essential for maintaining '
            'good health. Aim to drink at least eight glasses of water a day.\n\n3. Get enough sleep: '
            'Sleep is essential for  maintaining good health. Aim to get at least seven to eight hours of'
            ' quality sleep each night.'
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
        from inference import model_provider
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
            "The weather today is described as mostly sunny with a high temperature around 70 degrees "
            "Fahrenheit (21 degrees Celsius).\n\nTo determine if the weather will be similar tomorrow, "
            "you would need to check the weather forecast for tomorrow. The forecast may "
            "indicate similar weather conditions, or it may suggest different conditions such as rain or clouds."
            "\n\nTherefore, to answer your question, I would need to check the weather forecast for tomorrow. "
            "Once I have that information, I can tell you whether the weather is expected to be similar to today, "
            "or if it is expected to be different."
        ]
        expected_output_seq = "".join(expected_output)
        if torch.distributed.get_rank() == 0:
            similarity = self.edit_distance_similarity(output[:40], expected_output_seq[:40])
            print("similarity:", similarity)
            assert_judge(similarity > 0.75)
