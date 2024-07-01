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
    world_size = 2

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
        instruction = ["What are the characteristics of Suzhou?", "Introducing the Forbidden City in Beijing."]
        output = model.generate(instruction)
        expect_output1 = [
            "Suzhou is known for its beautiful gardens, canals, and classical architecture. It is also known for its silk industry and its cuisine. Suzhou is located in the Jiangsu province of China."
        ]
        expect_output2 = [
            'The Forbidden City is the largest and most complete ancient imperial palace complex in the world. It is a must-see attraction for any visitor to Beijing.'
            'The Forbidden City is located in the center of Beijing, and it is surrounded by a 52-meter-high wall and a 520-meter-long moat. The Forbidden City was built in the 15th century by the Ming Dynasty and was used as the imperial palace for 24 emperors until the end of the Qing Dynasty'
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
        instruction = "What are the characteristics of Hangzhou?"
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
            "Hangzhou has a long history and is famous for its beautiful scenery. It is also known for its rich cultural heritage. Hangzhou is located in the eastern part of Zhejiang Province, China. It is the capital of Hangzhou Prefecture-level City. Hangzhou has a subtropical monsoon climate with four distinct seasons. The average annual temperature is 16.2 °C (61.2 °F). The annual rainfall is 1,400 millimeters (55.1 in). Hangzhou has a humid subtropical climate with abundant rainfall and warm summers. The city is surrounded by mountains and rivers, making it a popular tourist destination. Hangzhou is known for its beautiful scenery, including West Lake, which is one of the most famous lakes in China. Hangzhou is also known for its rich cultural heritage, including the Longjing Tea Plantation, which is one of the most famous tea plantations in China. Hangzhou is a city with a long history and a rich cultural heritage. It has a subtropical monsoon climate with abundant rainfall and warm summers. It is also known for its beautiful scenery, including West Lake, and its rich cultural"
        ]
        expected_output_seq = "".join(expected_output)
        if torch.distributed.get_rank() == 0:
            similarity = self.edit_distance_similarity(output[:40], expected_output_seq[:40])
            print(output)
            print("similarity:", similarity)
            assert_judge(similarity > 0.75)
