# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import json
import os
import time
import gc
import argparse

import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from modelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig, Calibrator

WORKDIR = "./"
CEVAL_EXAM_DIR = os.path.join(WORKDIR, "ceval-exam")
SUBJECT_MAPPING_PATH = os.path.join(WORKDIR, "subject_mapping.json")
choices = ["A", "B", "C", "D"]
SEQ_LEN_IN = 32
SEQ_LEN_OUT = 32
SHOT = 5


class Record:
    def __init__(self, log_dir, log_flag=0):
        if not os.path.exists(log_dir):
            os.makedirs(name=log_dir, mode=0o750)
        self.log_dir = log_dir
        self.flag = log_flag
        self.log_name = os.path.join(log_dir, f"device{log_flag}.log")
        self.cache_name = os.path.join(log_dir, f"cache{log_flag}.csv")
        self.cache = self.load_cache()
    
    def log(self, *msg):
        with open(self.log_name, "a") as f:
            f.write(" ".join([str(i) for i in msg]) + '\n')
    
    def update_cache(self, task_name, question_id, truth_answer, predict_answer):
        with open(self.cache_name, "a") as f:
            f.write(f"{task_name},{question_id},{truth_answer},{predict_answer}\n")
        if task_name not in self.cache:
            self.cache[task_name] = 1
        else:
            self.cache[task_name] += 1
    
    def load_cache(self):
        if not os.path.exists(self.cache_name):
            self.log("[-] No cache file, cache will be created")
            return dict()
        self.log("[~] Loading cache on last abnormal exit ... (and continue with the cache)")
        with open(self.cache_name, "r") as f:
            cache = f.read().strip().split()
        if not cache:
            return dict()
        cache = [row.split(",") for row in cache]
        cache_dict = dict()
        tasks_name = set([t[0] for t in cache])
        for row in cache:
            if row[0] not in cache_dict:
                cache_dict[row[0]] = 1
            else:
                cache_dict[row[0]] += 1
        self.log(f"[+] Load cache successfully! {cache_dict}")
        return cache_dict


def init_model(args_in):
    init_model = AutoModelForCausalLM.from_pretrained(args_in.load_path,
                                                      trust_remote_code=True,
                                                      torch_dtype=torch.float32,
                                                      return_dict_in_generate=True,)
    init_tokenizer = AutoTokenizer.from_pretrained(args_in.load_path, use_fast=False, padding_side='left')
    
    init_model.eval()
    return init_model, init_tokenizer


def get_subject_mapping():
    with open(SUBJECT_MAPPING_PATH) as f:
        subject_mapping = json.load(f)
    return subject_mapping


def load_csv_by_task_name(task_name):
    dev_df = pd.read_csv(os.path.join(CEVAL_EXAM_DIR, "dev", task_name + "_dev.csv"), header=None)[:SHOT + 1]
    val_df = pd.read_csv(os.path.join(CEVAL_EXAM_DIR, "val", task_name + "_val.csv"), header=None)
    
    # remove the first row "column names" and the first column "id"
    dev_df = dev_df.iloc[1:, 1:]
    val_df = val_df.iloc[1:, 1:]
    return dev_df, val_df


def format_subject(subject):
    line = subject.split("_")
    sub = ""
    for entry in line:
        sub += " " + entry
    return sub


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = len(choices)
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def get_dataset(tokenizer_in, args_in):
    dataset_all = []
    task_name = 'college_economics'
    dev_df, val_df = load_csv_by_task_name(task_name)
    records = []
    for i in range(val_df.shape[0]):
        for cut_shot in range(SHOT):
            prompt_end = format_example(val_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task_name, SHOT - cut_shot)
            prompt = train_prompt + prompt_end
            input_len = len(tokenizer_in(prompt, return_tensors="pt").input_ids[0])
            if input_len > 2000:
                continue
            label = val_df.iloc[i, val_df.shape[1] - 1]
            records.append({'prompt': prompt, 'answer': label})
            break

    batch_size = args_in.batch_size
    for i in tqdm(range(0, len(records), batch_size)):
        end_idx = min(i + batch_size, len(records))
        prompt = [record['prompt'] for record in records[i: end_idx]]
        length = [len(record['prompt']) for record in records[i: end_idx]]
        inputs = tokenizer_in(prompt, return_tensors="pt", padding=True)
        inputs_calib = {}
        for k, v in inputs.items():
            inputs_calib[k] = v
        inputs_calib['max_new_tokens'] = 5
        dataset_all.append(inputs_calib)
    return dataset_all


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default="./llama2-7b",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--device",
        type=int,
        default="1",
        help="Run model on the specified devices",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="set batch_size",
    )
    parser.add_argument("--fraction", type=float, default=0.016)
    parser.add_argument("--amp_num", type=int, default=5)
    parser.add_argument("--act_method", type=int, default=3)
    parser.add_argument("--nonuniform", action="store_true")
    
    args = parser.parse_args()

    model, tokenizer = init_model(args)
    test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", max_length=SEQ_LEN_IN, truncation=True)
    with torch.no_grad():
        _ = model.generate(inputs_warm_up.input_ids, max_new_tokens=SEQ_LEN_OUT)

    # DO QUNAT
    dataset_calib = get_dataset(tokenizer, args)
    print('len of calib_dataset: ', len(dataset_calib))
    dataset_calib = dataset_calib[:]
    if args.batch_size == 2:
        print("bs = 2")

    mm_tensor = False
    fraction = args.fraction
    nonuniform = args.nonuniform
    quant_config = QuantConfig(w_bit=4,
                                    fraction=fraction,
                                    nonuniform=nonuniform,
                                    disable_names=['lm_head',
                                            'model.layers.0.self_attn.q_proj',
                                            'model.layers.0.self_attn.k_proj',
                                            'model.layers.0.self_attn.v_proj',
                                            'model.layers.0.self_attn.o_proj',
                                            'model.layers.0.mlp.gate_proj',
                                            'model.layers.0.mlp.up_proj',
                                            'model.layers.0.mlp.down_proj',
                                            'model.layers.1.self_attn.q_proj',
                                            'model.layers.1.self_attn.k_proj',
                                            'model.layers.1.self_attn.v_proj',
                                            'model.layers.1.self_attn.o_proj',
                                            'model.layers.1.mlp.gate_proj',
                                            'model.layers.1.mlp.up_proj',
                                            'model.layers.1.mlp.down_proj',
                                            'model.layers.2.self_attn.q_proj',
                                            'model.layers.2.self_attn.k_proj',
                                            'model.layers.2.self_attn.v_proj',
                                            'model.layers.2.self_attn.o_proj',
                                            'model.layers.2.mlp.gate_proj',
                                            'model.layers.2.mlp.up_proj',
                                            'model.layers.2.mlp.down_proj',
                                            'model.layers.4.self_attn.q_proj',
                                            'model.layers.4.self_attn.k_proj',
                                            'model.layers.4.self_attn.v_proj',
                                            'model.layers.4.self_attn.o_proj',
                                            'model.layers.4.mlp.gate_proj',
                                            'model.layers.4.mlp.up_proj',
                                            'model.layers.4.mlp.down_proj',
                                            'model.layers.30.self_attn.q_proj',
                                            'model.layers.30.self_attn.k_proj',
                                            'model.layers.30.self_attn.v_proj',
                                            'model.layers.30.self_attn.o_proj',
                                            'model.layers.30.mlp.gate_proj',
                                            'model.layers.30.mlp.up_proj',
                                            'model.layers.30.mlp.down_proj',
                                            ],
                                     act_method=3,
                                     mm_tensor=mm_tensor,
                                     co_sparse=True,
                                     pr=2.0)
    print('quant_config:', quant_config)
    print('config: ', quant_config.__dict__)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()
    model = calibrator.model
    
    dest_dir = './sparse_quant_llama_weight'
    if args.batch_size == 2:
        dest_path = os.path.join(dest_dir, 'bs2_amp5_llama2_' + str(args.fraction))
    else:
        dest_path = os.path.join(dest_dir, 'bs1_amp5_llama2_' + str(args.fraction))
    if mm_tensor:
        dest_path = os.path.join(dest_dir, 'llama2_per_tensor_' + str(args.fraction))
    calibrator.save(dest_path)
    print("time-cost: ", time.time() - start_time)

