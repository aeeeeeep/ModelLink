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

        if task == "beam_search_with_sampling":
            sys.argv = sys.argv + config.beam_search_with_sampling_auxliary_param
        elif task == "return_output_log_probs":
            sys.argv = sys.argv + config.return_output_log_probs_auxliary_param

        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
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


    def test_beam_search_with_sampling(self):
        """Beam Search with sampling"""
        self.init(config=ParamConfig, task="beam_search_with_sampling")
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )

        instruction = "Give me three tips for staying healthy."

        output = model.generate(
            instruction,
            num_beams=2,
            do_sample=True,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            max_new_tokens=self.args.max_new_tokens,
            tokenizer=None,
            stream=False
        )

        expect_output1 = '''1. Get enough sleep. A good night's sleep is important for your physical and mental health.\n2. Eat a balanced diet. Eating a variety of healthy foods can help you get the nutrients your body needs.\n3. Exercise regularly. Exercise can help you maintain a healthy weight, reduce stress, and improve your overall health.'''

        expect_output2 = '''Sure, here are three tips for staying healthy:\n1. Eat a balanced diet that includes fruits, vegetables, whole grains, and lean proteins.\n2. Get regular exercise, such as going for a walk or doing yoga.\n3. Get enough sleep each night, ideally 7-8 hours.'''

        if torch.distributed.get_rank() == 0:
            print(output)
            tokenizer = get_tokenizer()

            similarity1 = self.edit_distance_similarity(output[:30], expect_output1[:30])
            similarity2 = self.edit_distance_similarity(output[:30], expect_output2[:30])
            print("similarity1:", similarity1)
            print("similarity1:", similarity2)
            assert_judge(max(similarity1, similarity2) > 0.75)


    def test_return_output_log_probs(self):
        """Returns the probability distribution of tokens"""
        self.init(config=ParamConfig, task="return_output_log_probs")
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )

        instruction = "What is the whether like today?"

        output1, log_probs = model.generate(
            instruction,
            do_sample=True,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            max_new_tokens=self.args.max_new_tokens,
            tokenizer=None,
            stream=False,
            detokenize=False,
            return_output_log_probs=True
        )

        if torch.distributed.get_rank() == 0:
            tokenizer = get_tokenizer()
            print("--------------output1-------------")
            print(output1)
            print(tokenizer.decode(output1))

            expected_output1 = [2132, 686, 6761, 389, 1380, 498, 525, 304, 279, 1879,
                                13, 576, 9104, 646, 387, 2155, 304, 2155, 7482, 624,
                                872, 198, 3838, 374, 279, 9104, 1075, 304, 7148, 5267,
                                77091, 198, 785, 9104, 304, 7148, 3351, 374, 39698, 323]

            expected_output1_ext = [2132, 686, 6761, 389, 1380, 498, 525, 7407, 13, 16503,
                                    498, 3291, 752, 697, 3728, 5267, 872, 198, 29596, 11902,
                                    198, 77091, 198, 641, 9656, 11902, 11, 432, 594, 39698,
                                    3351, 13, 576, 9315, 374, 220, 23, 15, 12348, 68723]
            expected_output1_ext2 = [2132, 374, 83253, 16916,  3351, 382, 77091, 198, 3838, 374,
                                    279, 9104, 1075, 3351, 5267, 2610, 525, 264, 10950, 17847,
                                    13, 279, 198, 3838, 374, 279, 9104, 1075, 3351, 5267,
                                    2610, 525, 264, 10950, 17847, 13, 279, 198, 3838, 374]
            print("--------------log_probs----------------")
            print(log_probs.shape)
            assert_judge(log_probs.shape[0] == 256)
            assert_judge(log_probs.shape[1] == 151936)

            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output1[:40]).unsqueeze(0).float().npu(),
                                 output1[:40].unsqueeze(0).float())
            cos_sim = max(cos_sim, similarity(torch.tensor(expected_output1_ext[:40]).unsqueeze(0).float().npu(),
                                 output1[:40].unsqueeze(0).float()))
            cos_sim = max(cos_sim, similarity(torch.tensor(expected_output1_ext2[:40]).unsqueeze(0).float().npu(),
                                 output1[:40].unsqueeze(0).float()))
            print("similarity1: ", cos_sim)
            assert_judge(cos_sim > 0.75)