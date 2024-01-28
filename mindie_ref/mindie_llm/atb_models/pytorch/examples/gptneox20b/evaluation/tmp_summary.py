import json
import os

import pandas as pd

# CSV_PATH = "cache0_A100.csv"

CSV_PATH = "./test_result/cache0.csv"
WORKDIR = "./"
SUBJECT_MAPPING_PATH = os.path.join(WORKDIR, "subject_mapping.json")

HARD_TASK = (
    "advanced_mathematics", "discrete_mathematics", "probability_and_statistics", "college_chemistry",
    "college_physics",
    "high_school_mathematics", "high_school_chemistry", "high_school_physics")


def get_subject_mapping():
    with open(SUBJECT_MAPPING_PATH) as f:
        subject_mapping = json.load(f)
    return subject_mapping


def compute_metric(subject_mapping):
    run_results = pd.read_csv(CSV_PATH, names=['task_name', 'question_id', 'truth_answer', 'predict_answer'])
    classes_acc = dict()
    subject_acc = dict()
    hard_task = [0, 0]
    for task in subject_mapping:
        class_of_task = subject_mapping[task][2]
        this_task = run_results.loc[run_results.task_name == task]
        if not this_task.shape[0]:
            continue
        correct_num = (this_task.truth_answer == this_task.predict_answer).sum()
        if class_of_task not in classes_acc:
            classes_acc[class_of_task] = [0, 0]  # correct num, total num
        if task in HARD_TASK:
            hard_task[0] += correct_num
            hard_task[1] += this_task.shape[0]
        subject_acc[task] = correct_num / this_task.shape[0]
        classes_acc[class_of_task][0] += correct_num
        classes_acc[class_of_task][1] += this_task.shape[0]

    avg_acc = sum([i[0] for i in classes_acc.values()]) / sum([j[1] for j in classes_acc.values()])
    for c in classes_acc:
        classes_acc[c] = classes_acc[c][0] / classes_acc[c][1]
    classes_acc["Avg"] = avg_acc
    classes_acc["Avg(Hard)"] = hard_task[0] / hard_task[1]
    with open(f"tmp_subject_acc.json", "w") as fp:
        json.dump(subject_acc, fp, indent=2)
    with open(f"tmp_classes_acc.json", "w") as fp:
        json.dump(classes_acc, fp, indent=2)
    print(f"[+] Avg acc: {classes_acc['Avg']}")


subject_mapping = get_subject_mapping()
compute_metric(subject_mapping)
