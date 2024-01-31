import argparse
import json
import time
import os
from string import ascii_letters

import pandas as pd
import torch
import torch_npu
from tqdm import tqdm
from transformers import FalconForCausalLM, FalconConfig, FalconModel, AutoTokenizer

RESULT_OUTPUT_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], "test_result")
choices = ["A", "B", "C", "D"]
SEQ_LEN_IN = 32
SEQ_LEN_OUT = 32
SHOT = 5


class Record:
    def __init__(self, log_dir, log_flag):
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


def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch_npu.npu.set_device(local_rank)
    torch.manual_seed(1)
    return local_rank, world_size


def _init_torch_npu():
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)


def npu_load_model_falcon_multi_card(model_path, local_rank):
    ''' 当前进程只读取 local_rank 分片的权重，权重由切分脚本完成切分 '''
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

    tokenizer_path  = os.path.join(model_path, "tokenizer")
    part_model_path = os.path.join(model_path, "part_model", str(local_rank))

    time_start_load_model = time.time()
    print(f"[~] loading model ...", local_rank)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    model     = FalconForCausalLM.from_pretrained(part_model_path, pad_token_id=tokenizer.eos_token_id, torch_dtype=torch.float16).npu()
    print(f"[+] load model: {(time.time()-time_start_load_model)/60} min", local_rank)
    return model, tokenizer


def compute_metric(subject_mapping, recorder):
    run_results = pd.read_csv(recorder.cache_name, names=['task_name', 'question_id', 'truth_answer', 'predict_answer'])
    classes_acc = dict()
    subject_acc = dict()
    for task in subject_mapping:
        class_of_task = subject_mapping[task]
        this_task = run_results.loc[run_results.task_name == task]
        correct_num = (this_task.truth_answer == this_task.predict_answer).sum()
        if class_of_task not in classes_acc:
            classes_acc[class_of_task] = [0, 0]  # correct num, total num
        subject_acc[task] = correct_num / this_task.shape[0]
        classes_acc[class_of_task][0] += correct_num
        classes_acc[class_of_task][1] += this_task.shape[0]
        print(task, this_task.shape[0])
    avg_acc = sum([i[0] for i in classes_acc.values()]) / sum([j[1] for j in classes_acc.values()])
    for c in classes_acc:
        print(classes_acc[c][1])
        classes_acc[c] = classes_acc[c][0] / classes_acc[c][1]
    classes_acc["Avg"] = avg_acc
    with open(os.path.join(RESULT_OUTPUT_DIR, f"result_{recorder.flag}_subject_acc.json"), "w") as fp:
        json.dump(subject_acc, fp)
    with open(os.path.join(RESULT_OUTPUT_DIR, f"result_{recorder.flag}_classes_acc.json"), "w") as fp:
        json.dump(classes_acc, fp)


def get_subject_mapping():
    SUBJECT_MAPPING_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "subject_mapping_mmlu.json")
    with open(SUBJECT_MAPPING_PATH) as f:
        subject_mapping = json.load(f)
    return subject_mapping


def load_csv_by_task_name(task_name, dataset_path):
    dev_df = pd.read_csv(os.path.join(dataset_path, "dev", task_name + "_dev.csv"), header=None)[:SHOT + 1]
    val_df = pd.read_csv(os.path.join(dataset_path, "val", task_name + "_val.csv"), header=None)
    return dev_df, val_df


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


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


def batch_infer(model, tokenizer, records, recorder, task_name):
    start_idx = recorder.cache[task_name] if task_name in recorder.cache else 0
    for i in tqdm(range(start_idx, len(records))):
        prompt = records[i]['prompt']
        truth_answer = records[i]['answer']
        recorder.log("\n========== prompt start ==========\n", prompt, "\n==========  prompt end  ==========\n")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = len(inputs.input_ids[0])
        recorder.log(f"[+] prompt length: {input_len}")
        with torch.no_grad():
            output = model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(),
                                    max_new_tokens=SEQ_LEN_OUT)

        input_len = len(inputs.input_ids[0])

        answer = tokenizer.decode(output[0][input_len:]).strip()
        recorder.log("\n========== answer start ==========\n", answer, "\n==========  answer end  ==========\n")
        answer = [char.upper() for char in answer if char in ascii_letters]
        answer = answer[0] if answer else "-1"
        is_correct = "Correct" if answer == truth_answer else "Wrong"
        recorder.log(f"[{is_correct}] predict: {answer}, label: {truth_answer}")
        recorder.update_cache(task_name, i, truth_answer, answer)


def main(model, tokenizer, recorder, dataset_path):
    subject_mapping = get_subject_mapping()
    run_results = dict()
    model.eval()
    for task_name in subject_mapping:
        dev_df, val_df = load_csv_by_task_name(task_name, dataset_path)
        if task_name in recorder.cache and recorder.cache[task_name] == val_df.shape[0]:
            recorder.log(f"[~] Skip Task: {task_name}")
            continue
        recorder.log('[~] Testing %s ...' % task_name)
        records = []

        for i in range(val_df.shape[0]):
            k = SHOT
            for cut_shot in range(SHOT):
                prompt_end = format_example(val_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, task_name, SHOT - cut_shot)
                prompt = train_prompt + prompt_end
                input_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
                if input_len > 2000:
                    continue
                label = val_df.iloc[i, val_df.shape[1] - 1]
                records.append({'prompt': prompt, 'answer': label})
                break
        batch_infer(model, tokenizer, records, recorder, task_name)
    compute_metric(subject_mapping, recorder)


if __name__ == "__main__":
    _init_torch_npu()

    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--model_path",
        type=str,
        default = "/home/weights/falcon40b_4cards",
        help="Location of Model weights, which contains model folders",)
    parser.add_argument(
        "--dataset_path",
        default="/home/cy/datasets/mmlu/",
        help="dataset path",
    )
    args = parser.parse_args()
    print("args.model_path=",args.model_path)

    # Initialize parallel
    local_rank, world_size = setup_model_parallel()
    if not os.path.exists(RESULT_OUTPUT_DIR):
        try:
            os.makedirs(RESULT_OUTPUT_DIR)
        except:
            pass

    # Set recorder, log to different file by `local_rank`
    recorder = Record(RESULT_OUTPUT_DIR, local_rank)
    model, tokenizer = npu_load_model_falcon_multi_card(args.model_path, local_rank)

    recorder.log(f"[+] model device: {model.device}")
    recorder.log("--------------- warm up ---------------")
    test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", max_length=SEQ_LEN_IN, truncation=True)
    with torch.no_grad():
        _ = model.generate(inputs_warm_up.input_ids.npu(), max_new_tokens=SEQ_LEN_OUT)
    torch.npu.empty_cache()

    main(model, tokenizer, recorder, args.dataset_path)
