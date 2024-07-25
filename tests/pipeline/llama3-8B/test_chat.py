import sys
import os
import nltk
import torch
from torch import distributed as dist
import torch_npu
from tests.common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink
from megatron.legacy.model import GPTModel
from modellink.tasks.inference.text_generation.infer_base import add_text_generate_args, chat_get_instruction, chat_print_and_update_histories


class TestGeneration(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig, chat_type=None):
        """
        initialize the environment and arguments
        """
        sys.argv = [sys.argv[0]] + config.distributed_param + config.network_size + \
                   config.inference_param + config.auxiliary_param + config.tokenizer_param

        if chat_type == "hf_chat":
            sys.argv = sys.argv + config.inference_hf_chat_param
        elif chat_type == "prompt_chat":
            sys.argv = sys.argv + config.inference_prompt_chat_param

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


    def run_chat(self, model, turn0outputExpect):
        histories_no_template = []
        histories_template = []
        instruction = None

        test_questions = ["你能推荐几本深度学习的书吗？", "上面推荐的书建议学习顺序呢？", "9.11和9.9谁大？"]

        turns = 0
        while turns < 3:

            prompt = test_questions[turns]

            instruction = chat_get_instruction(self.args, histories_no_template, histories_template, prompt)

            responses = model.generate(
                instruction,
                do_sample=True,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                tokenizer=None,
                temperature=self.args.temperature,
                max_new_tokens=self.args.max_new_tokens,
                stream=True
            )
            output = chat_print_and_update_histories(self.args, responses, histories_no_template, histories_template, prompt)
            if torch.distributed.get_rank() == 0:
                print("-------------------------------")
                print(output)

                if(turns == 0):
                    similarity1 = self.edit_distance_similarity(output[:30], turn0outputExpect[0][:30])
                    similarity2 = self.edit_distance_similarity(output[:30], turn0outputExpect[1][:30])
                    print("similarity1:", similarity1)
                    print("similarity1:", similarity2)
                    assert_judge(max(similarity1, similarity2) > 0.75)

            turns = turns + 1


    def test_hf_chat(self):
        """Interactive dialog mode with multiple rounds of conversation"""
        self.init(config=ParamConfig, chat_type="hf_chat")
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )
        turn1outputExpect = []
        turn1outputExpect1 = "Here are some highly recommended books on deep learning that can help you dive deeper into the subject:"
        turn1outputExpect2 = '''Here are some highly recommended books for deep learning:\n\n**Foundational Books**\n\n1. **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**: This is the bible of deep learning.'''
        
        turn1outputExpect.append(turn1outputExpect1)
        turn1outputExpect.append(turn1outputExpect2)

        self.run_chat(model, turn1outputExpect)


    def test_prompt_type_chat(self):
        """Interactive dialog mode with multiple rounds of conversation"""
        self.init(config=ParamConfig, chat_type="prompt_chat")
        from inference import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )
        turn1outputExpect = []
        turn1outputExpect1 = "Here are some highly recommended books on deep learning that can help you dive deeper into the subject:"
        turn1outputExpect2 = '''Here are some highly recommended books for deep learning:\n\n**Foundational Books**\n\n1. **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**: This is the bible of deep learning.'''
        
        turn1outputExpect.append(turn1outputExpect1)
        turn1outputExpect.append(turn1outputExpect2)

        self.run_chat(model, turn1outputExpect)

