import sys
import os
import json
from pathlib import Path
import tqdm
import pandas as pd
import torch
import torch_npu
from transformers import AutoTokenizer
from tests.common import DistributedTest
from utils import ParamConfig, assert_judge
import modellink
from megatron.legacy.model import GPTModel
from modellink.tasks.evaluation.utils import add_text_generate_args


class TestEvaluation(DistributedTest):
    world_size = 8

    def init(self):
        sys.argv = [sys.argv[0]] + ParamConfig.network_size + \
                    ParamConfig.distributed_param + ParamConfig.evaluation_param + \
                    ParamConfig.regularization + ParamConfig.auxiliary_param + \
                    ParamConfig.inference_aux
                    
        from megatron.training.initialize import initialize_megatron
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})

        from megatron.training import get_args
        self.args = get_args()

    def test_mmlu_evaluation(self):
        self.init()
        from evaluation import model_provider
        from modellink.tasks.evaluation.eval_impl.template import MMLU_TEMPLATE_DIR
        model = GPTModel.from_pretrained(
            model_provider=model_provider,
            pretrained_name_or_path=self.args.load
        )
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, trust_remote_code=True)
        max_new_tokens = self.args.max_new_tokens

        instruction_template = "{few_shot_examples}\n\n{question}\nAnswer:"

        total_acc_n = 0
        total_n = 0

        test_dir = None
        for path in self.args.task_data_path:
            if "mmlu" in path:
                test_dir = path
        
        base_dir = Path(__file__).absolute().parent.parent.parent.parent
        template_dir = os.path.join(base_dir, MMLU_TEMPLATE_DIR)
        with open(template_dir, encoding='utf-8') as f:
            mmlu_few_shot_template = json.load(f)
   
        temp = []
        for file in tqdm.tqdm(os.listdir(test_dir)):
            file_path = os.path.join(test_dir, file)
            data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
            subject_name = file[0: -9]
            subject = subject_name.replace("_", " ")
            acc_n = 0
            data_df_test = data_df.iloc[0:15]
            for index, row in data_df_test.iterrows():
                test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                instruction = instruction_template.format(few_shot_examples=mmlu_few_shot_template[subject_name],
                                                          subject=subject,
                                                          question=test_question)
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
                    answer = chat_result[0].strip()
                    temp.append(answer)
                if answer == row['answer']:
                    acc_n += 1
            if torch.distributed.get_rank() == 0:
                total_n += len(data_df_test)
                total_acc_n += acc_n
        final_acc = 0
        if torch.distributed.get_rank() == 0:
            try:
                final_acc = total_acc_n / total_n
            except ZeroDivisionError as e:
                raise e
            print(final_acc)
            assert_judge(abs(final_acc - 0.265) <= 0.015)     

