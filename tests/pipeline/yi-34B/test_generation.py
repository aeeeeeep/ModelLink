import sys
import os
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
        instruction = ["春夏秋冬，四个季节"]
        output = model.generate(instruction, detokenize=False)
        expected_output1 = [101, 6001, 15831, 5074, 6435, 35308, 101, 31179, 44445, 60820,
                            11098, 60721, 8203, 61293, 60583, 35308, 102, 18024, 101, 59647,
                            60721, 60690, 60452, 4452, 59706, 60207, 24212, 1075, 61759, 60942,
                            63958, 60585, 59599, 21639, 101, 24212, 1075, 61287, 62566, 60632,
                            63011, 59599, 37835, 60408, 17664, 102, 60566, 9299, 49085, 101]
        expected_output2 = [101, 6001, 15831, 5074, 2045, 7753, 101, 5074, 2045, 8511,
                            102, 144, 18417, 101, 24018, 25592, 101, 59722, 60575, 59823,
                            39464, 60630, 59676, 59936, 59670, 101, 55550, 59688, 60686, 59801,
                            7292, 101, 60319, 60502, 60687, 61529, 101, 59722, 61418, 59632,
                            61441, 59936, 534, 448, 494, 534, 448, 494, 534, 455]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim1 = similarity(torch.tensor(expected_output1).unsqueeze(0).float().npu(),
                                 output[:50].unsqueeze(0).float())
            cos_sim2 = similarity(torch.tensor(expected_output2).unsqueeze(0).float().npu(),
                                 output[:50].unsqueeze(0).float())
            cos_sim = torch.max(cos_sim1, cos_sim2)
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
        expected_output = [39047, 59654, 101, 79, 77, 77, 85, 59867, 10536, 60397,
                           536, 493, 487, 59732, 60516, 101, 62731, 62131, 59757, 59637,
                           59635, 60382, 59689, 35444, 59670, 101, 59630, 61004, 60159, 60475,
                           59638, 101, 6919, 59678, 2349, 11923, 17463, 60243, 60034, 59652,
                           22740, 59599, 9034, 102, 144, 79, 77, 77, 85, 59867]

        if torch.distributed.get_rank() == 0:
            print(output)
            similarity = torch.nn.CosineSimilarity(dim=1)
            cos_sim = similarity(torch.tensor(expected_output).unsqueeze(0).float().npu(),
                                 output[:50].unsqueeze(0).float())
            print("similarity: ", cos_sim)
            assert_judge(cos_sim > 0.95)
