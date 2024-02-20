#!/usr/bin/env python
# _*_coding:utf-8_*_
"""
@Time   :  2024/2/9 14:46
@Author :  Qinghua Wang
@Email  :  597935261@qq.com
"""
import os
from unittest import TestCase

import torch
import torch.nn as nn

os.environ["TIMEIT"] = "1"
from atb_speed.common.timer import Timer


class AddNet(nn.Module):
    def __init__(self, in_dim, h_dim=5, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)

    @Timer.timing
    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class TimerTest(TestCase):
    @classmethod
    def setUpClass(cls):
        Timer.reset()
        # Timer.sync= xxxx
        cls.add_net = AddNet(in_dim=2)

    def test_1(self):
        for i in range(5):
            x = torch.randn(1, 1)
            y = torch.randn(1, 1)
            result = self.add_net.forward(x, y)
            print(result)
        print(Timer.timeit_res)
        print(Timer.timeit_res.first_token_delay)
        print(Timer.timeit_res.next_token_avg_delay)
