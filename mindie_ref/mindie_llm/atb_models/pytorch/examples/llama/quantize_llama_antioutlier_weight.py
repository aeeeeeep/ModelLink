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
import os
import argparse
import json
import pandas as pd
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
import tqdm

current_path = os.getcwd()

SHOT = 5
choices = ["A", "B", "C", "D"]

SEQ_LEN_OUT = 32


def load_csv_by_task_name(task_name):
    ceval_path = os.path.join(current_path, 'ceval-exam')
    dev_df = pd.read_csv(os.path.join(ceval_path, 'dev', task_name + "_dev.csv"), header=None)[:SHOT + 1]
    val_df = pd.read_csv(os.path.join(ceval_path, 'val', task_name + "_val.csv"), header=None)
    
    # remove the first row "column names" and the first column "id"
    dev_df = dev_df.iloc[1:, 1:]
    val_df = val_df.iloc[1:, 1:]
    return dev_df, val_df


def get_dataset(args_in, tokenizer_in):
    dataset_all = []
    task_name = args_in.quant_task_name
    print("task name", task_name)
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

    for i in range(0, len(records)):
        prompt = records[i]['prompt']
        inputs = tokenizer_in(prompt, return_tensors="pt")

        inputs_calib = {}
        for k, v in inputs.items():
            inputs_calib[k] = v.cpu()

        dataset_all.append(inputs_calib)
    return dataset_all


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


def format_subject(subject):
    line = subject.split("_")
    sub = ""
    for entry in line:
        sub += " " + entry
    return sub   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="antioutlier quantizing Model weights generate.")
    parser.add_argument(
        "--input_path",
        default="/data/llama-2/Llama-2-13b-hf",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/data/models/llama2-7b-int8_model',
        help="Location to write the weights",
    )
    parser.add_argument(
        "--model_type",
        default="Llama",
        help="the type of the model in AntiOutlier config",
    )
    parser.add_argument(
        "--disable_level",
        default='L8',
        help="number of layers that don't need to be quantized",
    )
    parser.add_argument(
        "--disable_idx_lst",
        default=['0'],
        help="layers that don't need to be quantized",
    )
    parser.add_argument(
        "--quant_task_name",
        default='teacher_qualification',
        help="name of the task in the quantizing dataset",
    )
    
    args = parser.parse_args()
    # load float model weight
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.input_path,
                                            trust_remote_code=True)
    dataset_calib = get_dataset(args, tokenizer)
    print('len of calib_dataset: ', len(dataset_calib))
    dataset_calib = dataset_calib[:]

    temp_calib_data = []
    for item in dataset_calib:
        temp_calib_data.append([item["input_ids"], item["attention_mask"]])
    
    disable_idx_lst = [int(idx) for idx in args.disable_idx_lst.strip('[').strip(']').split(",")]
    disable_names = []
    for layer_index in disable_idx_lst:
        q_pack_name = "model.layers.{}.self_attn.q_proj".format(layer_index)
        k_pack_name = "model.layers.{}.self_attn.k_proj".format(layer_index)
        v_pack_name = "model.layers.{}.self_attn.v_proj".format(layer_index)
        o_proj_name = "model.layers.{}.self_attn.o_proj".format(layer_index)
        up_proj_name = "model.layers.{}.mlp.up_proj".format(layer_index)
        gate_proj_name = "model.layers.{}.mlp.gate_proj".format(layer_index)
        down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
        disable_names.append(q_pack_name)
        disable_names.append(k_pack_name)
        disable_names.append(v_pack_name)
        disable_names.append(o_proj_name)
        disable_names.append(up_proj_name)
        disable_names.append(gate_proj_name)
        disable_names.append(down_proj_name)
    disable_names.append("lm_head")

    # 离群值抑制
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.input_path,
                                                torch_dtype=torch.float32, trust_remote_code=True).cpu()
    model.eval()

    print("antioutlier suppression start")
    anti_config = AntiOutlierConfig(anti_method="m2", dev_type="cpu")
    anti_outlier = AntiOutlier(model, calib_data=temp_calib_data, cfg=anti_config, model_type=args.model_type)
    anti_outlier.process()
    print("antioutlier suppression success")

    # 保存antioutlier浮点权重
    anti_weight_path = os.path.join(args.output_path, 'anti_weight')
    model.save_pretrained(anti_weight_path)
    print(f'Save antioutlier float weight in {anti_weight_path} success')

    # 量化
    quant_config = QuantConfig(w_bit=8, disable_names=disable_names,
                            dev_type='cpu', act_method=1, pr=1.0, mm_tensor=False)

    calibrator = Calibrator(model, quant_config, calib_data=temp_calib_data, disable_level=args.disable_level)
    print('quantilize start')
    calibrator.run()
    print('quantilize success')

    # 保存antioutlier量化权重
    calibrator.save(args.output_path)
    print(f'Save quant weight in {args.output_path} success')