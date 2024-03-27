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
import os
import logging
import json
import pandas as pd
import tqdm

from .template import BBH_TEMPLATE_DIR
from ..eval_api.dataset_eval import DatasetEval
from ..eval_api.chat import Chat
from ....error_utils import check_divisible_by_zero

logger = logging.getLogger(__name__)


class BBHEval(DatasetEval):
    def __init__(self, test_dir, batch_size,
                 instruction_template="{fewshot_template}Q: {question}\nA:"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.batch_size = batch_size

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        total_acc_n = 0
        total_n = 0
        score_datas = []
        sample_n = 0
        rank = None
        with open(BBH_TEMPLATE_DIR, encoding='utf-8') as f:
            bbh_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            with open(file_path, encoding='utf-8') as f:
                bbh_dataset = json.load(f)
            subject_name = file[0: -5]
            subject_result = {}
            sample_n += len(bbh_dataset['examples'])
            acc_n = 0
            sorted_dataset = sorted(bbh_dataset['examples'], key=lambda x: len(x['input']))
            instructions = []
            targets = []
            for idx, item in enumerate(sorted_dataset):
                instruction = self.instruction_template.format(fewshot_template=bbh_template[subject_name],
                                                               question=item['input'])
                instructions.append(instruction)
                targets.append(item['target'])

                if len(instructions) == self.batch_size or len(bbh_dataset['examples']) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            try:
                                if rank == 0:
                                    logger.info("correct: %s, AI: %s", targets[index], answer.splitlines()[0])
                                    subject_result[str(idx - len(chat_results) + index + 1)] = answer.splitlines()[0]
                                    if subject_result[str(idx - len(chat_results) + index + 1)] == targets[index]:
                                        acc_n += 1
                            except Exception as e:
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + f". AI answer: {answer}"
                    instructions = []
                    targets = []

            if rank == 0:
                logging.info(f"{subject_name} acc = {acc_n}/{len(bbh_dataset['examples'])}="
                             f"{check_divisible_by_zero(acc_n, len(bbh_dataset['examples']))}")
                total_n += len(bbh_dataset['examples'])
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(bbh_dataset['examples']),
                                    check_divisible_by_zero(acc_n, len(bbh_dataset['examples']))])
        if rank == 0:
            logger.info(f"bbh acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, check_divisible_by_zero(total_acc_n, total_n)])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
