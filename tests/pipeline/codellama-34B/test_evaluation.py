import sys
import os
import json
import logging
import subprocess
import torch
import tqdm
import torch_npu
from transformers import AutoTokenizer
from ascendspeed import megatron_adaptor
from common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink
from megatron.model import GPTModel
from modellink.tasks.evaluation.utils import add_text_generate_args
from modellink.tasks.evaluation.eval_impl.human_eval import extract_answer_code


class TestEvaluation(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig):
        sys.argv = [sys.argv[0]] + config.distributed_param + config.network_size + \
                   config.evaluation_param + config.auxiliary_param + config.tokenizer_param
        from megatron.initialize import initialize_megatron
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
        from megatron import get_args
        self.args = get_args()
        self.test_dir = self.args.task_data_path[0]

    def read_problems(self):
        return {task["task_id"]: task for task in self.stream_jsonl(self.test_dir)}

    def stream_jsonl(self, test_dir: str):
        """
        Parses each jsonl line and yields it as a dictionary
        """
        for file in os.listdir(test_dir):
            test_code_path = os.path.join(self.test_dir, file)
            with open(test_code_path, 'r') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)

    def test_human_eval_evaluation(self):
        self.init(config=ParamConfig)
        from evaluation import model_provider
        logger = logging.getLogger(__name__)
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, trust_remote_code=True)
        max_new_tokens = self.args.max_new_tokens

        instruction_template = "{prompt}"
        success_n = 0
        problems = self.read_problems()
        problems_num = 11
        for idx, (task_id, task) in enumerate(problems.items()):
            if idx == problems_num:
                break
            instruction = instruction_template.format(prompt=task['prompt'])
            chat_result = model.generate(
                instruction,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
                stream=False,
                return_output_log_probs=True
            )
            assert_judge(isinstance(chat_result, tuple))
            assert_judge(isinstance(chat_result[1], torch.Tensor))
            answer = None
            if chat_result:
                answer = chat_result[0].lstrip()
            try:
                if torch.distributed.get_rank() == 0:
                    python_execute = sys.executable
                    answer = task['prompt'] + "    " + answer
                    test_file = extract_answer_code(answer, task)
                    result = subprocess.run([python_execute, test_file], capture_output=True, timeout=10)
                    if result.returncode != 0:
                        error_msg = result.stderr.decode("utf-8")
                        logger.info(error_msg)
                    else:
                        success_n += 1
            except Exception as e:
                if torch.distributed.get_rank() == 0:
                    logger.info("%s failed. %s", task_id, e)
            finally:
                pass
        
        if torch.distributed.get_rank() == 0:
            try:
                final_acc = success_n / problems_num
            except ZeroDivisionError as e:
                raise e
            print(final_acc)
            assert_judge(abs(final_acc - 0.45) < 0.01)