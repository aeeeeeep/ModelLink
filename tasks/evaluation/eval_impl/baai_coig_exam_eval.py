# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION. All rights reserved.
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
from ascendspeed.error_utils import check_divisible_by_zero
from tasks.evaluation.eval_api.dataset_eval import DatasetEval
from tasks.evaluation.eval_api.llm_chat import LlmChat

logger = logging.getLogger(__name__)


class BaaiCoigEval(DatasetEval):
    def __init__(self, test_dir,
                 instruction_template="{fewshot_template}\n\n问：{question}\n答："):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.file_context = None
        self.test_case_list = dict()
        self.answer_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def test_case_process(self, test_case):
        test_question = test_case['textbox_question']
        textbox_q_context = ''
        if 'textbox_q_context' in test_case and len(test_case['textbox_q_context']) > 0:
            textbox_q_context = test_case['textbox_q_context']
        for item in self.answer_list:
            test_question = test_question.replace(f' {item}.', f' \n{item}.')
        suggest_head = "请从下列选择题中选择正确的选项。\n\n问："
        return f"{textbox_q_context}\n{suggest_head}{test_question}\n答："

    def test_case_select(self, item):
        subject = item["subject"]
        if item["textbox_answer"] in self.answer_list:
            if subject not in self.test_case_list:
                self.test_case_list[subject] = []
            self.test_case_list[subject].append(item)

    def get_test_case_list(self, file_path):
        with open(file_path, "r") as files:
            for item in files:
                data = json.loads(item)
                self.test_case_select(data)

    def eval(self, llm_chat: LlmChat) -> (dict, pd.DataFrame):
        answer_result = {}
        match_count_total = 0
        case_count_total = 0
        score_datas = []
        rank = None
        file_name_list = os.listdir(self.test_dir)
        if not file_name_list:
            return None, None
        file_name = file_name_list[0]
        file_path = os.path.join(self.test_dir, file_name)
        self.get_test_case_list(file_path)
        for subject_name in self.test_case_list:
            test_cases = self.test_case_list[subject_name]
            subject_result = {}
            match_count = 0
            for idx, test_case in enumerate(test_cases):
                test_question = self.test_case_process(test_case)
                chat_result, rank = llm_chat.chat(instruction=test_question, history=[])
                answer = chat_result[0]
                if rank == 0:
                    subject_result[str(idx)] = answer
                if rank == 0 and subject_result[str(idx)] == test_case['textbox_answer']:
                    match_count += 1

            if rank == 0:
                acc = check_divisible_by_zero(match_count, len(test_cases))
                logging.info("%s acc = %d/%d=%e",
                             subject_name, match_count, len(test_cases), acc)
                case_count_total += len(test_cases)
                match_count_total += match_count
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(test_cases), acc])

        if rank == 0:
            total_acc = check_divisible_by_zero(match_count_total, case_count_total)
            logging.info("ceval acc = %d/%d=%e", match_count_total, case_count_total, total_acc)
            score_datas.append(["total", case_count_total, total_acc])

        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
