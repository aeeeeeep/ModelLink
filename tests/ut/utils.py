"""
We can't use assert in our code for codecheck, so create this auxiliary function to wrap
the assert case in ut for ci.
"""
import os
import glob
import torch


def judge_expression(expression):
    if not expression:
        raise AssertionError


def compare_state_dicts(state_dict1, state_dict2):
    if state_dict1.keys() != state_dict2.keys():
        return False

    for key in state_dict1.keys():
        value1 = state_dict1[key]
        value2 = state_dict2[key]

        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            if not torch.equal(value1, value2):
                print(f"Difference found in key: {key}")
                return False
        elif isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_state_dicts(value1, value2):
                return False
        else:
            pass

    return True


def weight_compare(dir_1, dir_2, suffix="pt"):
    models_path = glob.glob(os.path.join(dir_1, '**', f'*.{suffix}'), recursive=True)
    for path_1 in models_path:
        path_2 = path_1.replace(dir_1, dir_2)
        state_dict1 = torch.load(path_1)
        state_dict2 = torch.load(path_2)
        are_equal = compare_state_dicts(state_dict1, state_dict2)
        if not are_equal:
            return False

    return True
