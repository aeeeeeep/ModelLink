import sys
import os
import torch
import nltk
from tests.common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink
from megatron.legacy.model import GPTModel
from megatron.training import get_args, get_tokenizer
from megatron.training.initialize import initialize_megatron
from modellink.tasks.inference.text_generation.infer_base import add_text_generate_args


class TestGeneration(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig, task=None):
        """
        initialize the environment and arguments
        """
        sys.argv = [sys.argv[0]] + config.distributed_param_tp8_pp1 + config.network_size + \
                config.inference_param + config.beam_search_auxliary_param + config.auxiliary_param + config.tokenizer_param

        if task == "beam_search":
            sys.argv = sys.argv + config.beam_search_auxliary_param
        elif task == "greedy":
            sys.argv = sys.argv + config.greedy_search_auxliary_param
        elif task == "do_sample":
            sys.argv = sys.argv + config.do_sample_auxliary_param

        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
        self.args = get_args()


    def test_beam_search(self):
        """
        load weight to get model and construct the prompts to generate output, 
        and compare with expected for `beam search`.
        """
        self.init(config=ParamConfig, task="beam_search")
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )

        max_new_tokens = self.args.max_new_tokens
        instruction = "如何提高身体素质"
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

        if torch.distributed.get_rank() == 0:
            print("----------------------output-------------------------")
            print(output)
            expected_output1 = [100627, 101099, 100838, 104339, 101194, 3837, 87752, 99639, 6684, 31338,
                                96422, 28311, 16, 13, 4891, 251, 248, 68878, 101079, 5122,
                                106854, 104102, 71817, 16, 20, 15, 83031, 9370, 15946, 49567,
                                102660, 18830, 100316, 101079, 3837, 29524, 99234, 99314, 5373, 107530]

            expected_output2 = [30534, 100627, 101099, 100838, 3837, 73670, 103975, 87752, 101082, 28311,
                                16, 13, 4891, 223, 98, 99446, 104579, 5122, 101907, 109635,
                                103170, 107151, 5373, 100912, 52510, 116570, 5373, 105349, 5373, 105373,
                                33108, 117094, 49567, 102100, 101252, 3837, 101153, 44636, 108461, 5373]

            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output1).unsqueeze(0).float().npu(),
                                 output[:40].unsqueeze(0).float())
            cos_sim = max(cos_sim, similarity(torch.tensor(expected_output2).unsqueeze(0).float().npu(),
                                 output[:40].unsqueeze(0).float()))
            print("similarity: ", cos_sim)
            assert_judge(cos_sim > 0.85)


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
        self.init(config=ParamConfig, task="greedy")
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )

        instruction = ["What are the characteristics of Suzhou?", "Introducing the Forbidden City in Beijing."]
        output = model.generate(instruction)
        expect_output1 = [
            "Suzhou is a city in China. It is known for its beautiful gardens, canals, and classical Chinese architecture. It is also known for its silk production and traditional arts and crafts. The city has a rich cultural heritage and is home to many historic temples and museums. Additionally, Suzhou is known for its cuisine, which features local specialties such as sweet and sour fish and rice cakes."
        ]
        expect_output2 = [
            'The Forbidden City is a palace complex in Beijing, China. It was the home of the emperors of China for almost 500 years, from the Ming Dynasty to the end of the Qing Dynasty. The complex covers an area of 72 hectares and has over 9,000 rooms. It is a UNESCO World Heritage Site and one of the most popular tourist attractions in China..'
        ]

        expect_output1_seq = "".join(expect_output1)
        expect_output2_seq = ''.join(expect_output2)
        
        if torch.distributed.get_rank() == 0:
            print("----------------------output1-------------------------")
            print(output[0])
            print("----------------------output2-------------------------")
            print(output[1])
            
            similarity1 = self.edit_distance_similarity(output[0][:30], expect_output1_seq[:30])
            similarity2 = self.edit_distance_similarity(output[1][:30], expect_output2_seq[:30])
            print("similarity1:", similarity1)
            print("similarity2:", similarity2)
            assert_judge(similarity1 > 0.85)
            assert_judge(similarity2 > 0.85)


    def test_do_sample(self):
        """Do Sample"""
        self.init(config=ParamConfig, task="do_sample")
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )

        instruction = "what is Disneyland?"

        output = model.generate(
            [instruction, instruction],
            do_sample=True,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            max_new_tokens=self.args.max_new_tokens,
            tokenizer=None,
            stream=False
        )

        expect_output1 = "Disneyland Park is an entertainment park located in Anaheim, California, United States. It is owned by the Disney Parks, Experiences and Consumer Products division of the American multinational conglomerate corporation the Walt Disney Company. It is also the first of seven theme parks built at Walt Disney's original vision, where visitors can enjoy various attractions, entertainment, and dining."
        expect_output1_seq = "".join(expect_output1)

        if torch.distributed.get_rank() == 0:
            print(output)
            tokenizer = get_tokenizer()

            similarity1 = self.edit_distance_similarity(output[0][:30], expect_output1_seq[:30])
            print("similarity1:", similarity1)
            assert_judge(similarity1 > 0.85)