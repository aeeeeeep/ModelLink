import sys
import os
import json
import torch
import tqdm
import torch_npu
from utils import ParamConfig, assert_judge
from transformers import AutoTokenizer
from ascendspeed import megatron_adaptor
import modellink
from megatron.model import GPTModel
from tests.pipeline.common import DistributedTest
from tasks.evaluation.evaluation_llama import add_text_generate_args


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

    def get_result(self, tokenizer, result):
        if result:
            final_result = [result[0]]
            if result[1][0][tokenizer.encode("Yes")[-1]] >= result[1][0][tokenizer.encode("No")[-1]]:
                final_result.append("T")
            else:
                final_result.append("F")
        else:
            final_result = None
        return final_result

    def test_boolq_evaluation(self):
        self.init(config=ParamConfig)
        from tasks.evaluation.evaluation_llama import model_provider
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=self.args.load
        )
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, trust_remote_code=True)
        max_new_tokens = self.args.max_new_tokens

        instruction_template = "{passage}\nQuestion: {question}?\nAnswer:"

        answer_result = {}
        total_acc_n = 0
        total_n = 0
        test_dir = None
        for path in self.args.task_data_path:
            if "boolq" in path:
                test_dir = path
        for file in tqdm.tqdm(os.listdir(test_dir)):
            file_path = os.path.join(test_dir, file)
            with open(file_path, encoding='utf-8') as f:
                boolq_question_list = []
                for line in f.readlines():
                    boolq_question_list.append(json.loads(line))
            boolq_question_list = boolq_question_list[:60]
            subject_result = {}
            acc_n = 0
            for index, item in enumerate(boolq_question_list):
                instruction = instruction_template.format(passage=item['passage'], question=item['question'])
                result = model.generate(
                    instruction,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    tokenizer=tokenizer,
                    stream=False,
                    return_output_log_probs=True
                )
                result = self.get_result(tokenizer, result)
                if result:
                    answer = result[1]
                else:
                    answer = None
                try:
                    if torch.distributed.get_rank() == 0:
                        subject_result[str(index)] = answer
                        if subject_result[str(index)] == str(item['answer'])[0]:
                            acc_n += 1
                except Exception as e:
                    raise e
            if torch.distributed.get_rank() == 0:
                total_n += len(boolq_question_list)
                total_acc_n += acc_n
                answer_result['Boolq_dataset'] = subject_result
        if torch.distributed.get_rank() == 0:
            try:
                final_acc = total_acc_n / total_n
            except ZeroDivisionError as e:
                raise e
            print(final_acc)
            assert_judge(abs(final_acc - 0.71) < 0.03)
