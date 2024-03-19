# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import copy
import math
import os
import time

import torch
import json
import random
from dataclasses import dataclass
from torch import nn

@dataclass
class next_token_param:
    random_seed: int
    temperature:float
    top_k: int
    top_p: float
    min_tokens_to_keep: int
    def __post_init__(self):
        self.temperature = torch.HalfTensor([self.temperature]).npu()
        self.top_p = torch.HalfTensor([self.top_p]).npu()

def sample_search(test_data, logits: torch.Tensor):
    topktopp_op_name = torch.classes.OperationTorch.OperationTorch("Layerstopktopp")
    topktopp_param = json.dumps({"axes": 1, "headNum": 0, "topk": test_data.top_k, "vocsize": logits.size()[1], "row":logits.size()[0], "randseed":test_data.random_seed, "min_tokens_to_keep": test_data.min_tokens_to_keep})
    topktopp_op_name.set_param(topktopp_param)
    next_logits, next_token=topktopp_op_name.execute([logits, test_data.top_p, test_data.temperature])
    return next_token