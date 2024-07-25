# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate LLAMA"""
import os
import sys
import time
import logging
from torch import distributed as dist
from transformers import AutoTokenizer
import modellink
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.legacy.model import GPTModel
from megatron.training.arguments import core_transformer_config_from_args

from modellink.tasks.evaluation.utils import add_text_generate_args
from modellink.tasks.evaluation.eval_api.chat import Chat
from modellink.tasks.evaluation.eval_impl.boolq_eval import BoolqEval
from modellink.tasks.evaluation.eval_impl.gsm8k_eval import Gsm8kEval
from modellink.tasks.evaluation.eval_impl.mmlu_eval import MmluEval
from modellink.tasks.evaluation.eval_impl.ceval_exam import CEvalExam
from modellink.tasks.evaluation.eval_impl.bbh_eval import BBHEval
from modellink.tasks.evaluation.eval_impl.agi_eval import AGIEvalExam
from modellink.tasks.evaluation.eval_impl.human_eval import HumanEval

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def model_provider(pre_process=True, post_process=True):
    config = core_transformer_config_from_args(get_args())

    """Build the model."""
    init_model = GPTModel(
        config,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )
    return init_model


def get_result(result, tokenizer):
    if result:
        final_results = []
        if isinstance(result[0], list):
            for idx, res in enumerate(result[0]):
                final_result = [res]
                if result[1][idx][0][tokenizer.encode("Yes")[-1]] >= result[1][idx][0][tokenizer.encode("No")[-1]]:
                    final_result.append('T')
                else:
                    final_result.append('F')
                final_results.append(final_result)
        else:
            final_result = [result[0]]
            if result[1][0][tokenizer.encode("Yes")[-1]] >= result[1][0][tokenizer.encode("No")[-1]]:
                final_result.append('T')
            else:
                final_result.append('F')
            final_results.append(final_result)
    else:
        final_results = None
    return final_results


class LLMChat(Chat):
    def __init__(self, llm_args, model, tokenizer):
        self.args = llm_args
        self.model = model
        self.tokenizer = tokenizer
        self.template = "{instruction}"

    def chat(self, instruction, history):
        instruction_temp = None
        if self.args.prompt_type is None:
            instruction_temp = [self.template.format(instruction=ins) if (self.tokenizer.chat_template is None or self.args.no_chat_template) else self.tokenizer.apply_chat_template([{"role": "user", "content": ins}]) for ins in instruction]
        else:
            instruction_temp = instruction

        result = self.model.generate(
            instruction_temp,
            do_sample=False,
            max_new_tokens=self.args.max_new_tokens,
            stream=False,
            return_output_log_probs=True
        )
        return get_result(result, self.tokenizer), dist.get_rank()

    def beam_search_chat(self, instruction, history):
        instruction_temp = None
        if self.args.prompt_type is None:
            instruction_temp = self.template.format(instruction=instruction) if (self.tokenizer.chat_template is None or self.args.no_chat_template) else self.tokenizer.apply_chat_template([{"role": "user", "content": instruction}])
        else:
            instruction_temp = instruction

        result = self.model.generate(
            instruction_temp,
            do_sample=False,
            max_new_tokens=self.args.max_new_tokens,
            stream=False,
            num_beams=4,
            top_k=50,
            top_p=0.95,
            length_penalty=0.7
        )
        return [result], dist.get_rank()


def mmlu(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None
    for path in eval_args.task_data_path:
        if 'mmlu' in path:
            data_path = path
    try:
        if data_path:
            mmlu_eval = MmluEval(test_dir=data_path, eval_args=eval_args)
            answer, score_df = mmlu_eval.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def gsm8k(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None
    for path in eval_args.task_data_path:
        if 'gsm8k' in path:
            data_path = path
    try:
        if data_path:
            gsm8k_eval = Gsm8kEval(test_dir=data_path, batch_size=eval_args.evaluation_batch_size)
            answer, score_df = gsm8k_eval.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def boolq(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        if 'boolq' in path:
            data_path = path
    try:
        if data_path:
            boolq_eval = BoolqEval(test_dir=data_path, eval_args=eval_args)
            answer, score_df = boolq_eval.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def ceval(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        if 'ceval' in path:
            data_path = path
    try:
        if data_path:
            ceval_exam = CEvalExam(test_dir=data_path, eval_args=eval_args)
            answer, score_df = ceval_exam.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def human_eval(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        if 'human_eval' in path:
            data_path = path
    try:
        if data_path:
            human_eval_exam = HumanEval(test_dir=data_path, instruction_template=eval_args.instruction_template)
            answer, score_df = human_eval_exam.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def agi_eval(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        if 'agieval' in path:
            data_path = path
    try:
        if data_path:
            agieval_exam = AGIEvalExam(test_dir=data_path, batch_size=eval_args.evaluation_batch_size)
            answer, score_df = agieval_exam.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def bbh_eval(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        if 'bbh' in path:
            data_path = path
    try:
        if data_path:
            bbh = BBHEval(test_dir=data_path, batch_size=eval_args.evaluation_batch_size)
            answer, score_df = bbh.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def main():
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()
    model = GPTModel.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)

    rank = dist.get_rank()
    if 'mmlu' in args.task:
        a = time.time()
        mmlu(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'MMLU Running Time:, {time.time() - a}')
    if 'gsm8k' in args.task:
        a = time.time()
        gsm8k(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'GSM8k Running Time: {time.time() - a}')
    if 'boolq' in args.task:
        a = time.time()
        boolq(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'Boolq Running Time: {time.time() - a}')
    if 'ceval' in args.task:
        a = time.time()
        ceval(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'Ceval Running Time: {time.time() - a}')
    if 'bbh' in args.task:
        a = time.time()
        bbh_eval(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'bbh Running Time: {time.time() - a}')
    if 'agieval' in args.task:
        a = time.time()
        agi_eval(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'agi_eval Running Time: {time.time() - a}')
    if 'human_eval' in args.task:
        a = time.time()
        human_eval(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'Human_eval Running Time: {time.time() - a}')


if __name__ == "__main__":
    main()

