# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
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

import unittest
import json
import torch
from wrapt_timeout_decorator import timeout
from ascendspeed.core import parallel_state
from ascendspeed.core.memory.auto_recomputing.autorecompute_solver import solve_graph


class TestAutoRecomputing(unittest.TestCase):
    @timeout(1200)
    def test_function(self):
        str1 = '''
    {
        "module": [],
        "free_mem": 0,
        "layers": [{
            "name": "module",
            "layers": [
            {
                "name": "module",
                "layers": [
                {
                    "name": "embedding",
                    "layers": [
                    {
                        "name": "word_embeddings",
                        "memory": 256,
                        "time": 13.043999671936035,
                        "input": 0.25,
                        "peak_memory": 268797952,
                        "forward_cnt": 2,
                        "pre_total_time": 26.08799934387207
                    }],
                    "memory": 64,
                    "time": 16.85166358947754,
                    "input": 0.25,
                    "peak_memory": 604310016,
                    "forward_cnt": 2,
                    "pre_total_time": 33.70332717895508
                }, 
                {
                    "name": "language_model",
                    "layers": [
                    {
                        "name": "layers",
                        "layers": [
                        {
                            "name": "0",
                            "layers": [
                            {
                                "name": "input_layernorm",
                                "memory": 384,
                                "time": 1.9710063934326172,
                                "input": 64.0,
                                "peak_memory": 402705408,
                                "forward_cnt": 2,
                                "pre_total_time": 3.9420127868652344
                            }, {
                                "name": "attention",
                                "layers": [{
                                    "name": "query_key_value",
                                    "memory": 192,
                                    "time": 9.331226348876953,
                                    "input": 64.0,
                                    "peak_memory": 402654208,
                                    "forward_cnt": 2,
                                    "pre_total_time": 18.662452697753906
                                }, {
                                    "name": "rotary_emb",
                                    "memory": 0,
                                    "time": 1.7354488372802734,
                                    "input": 64.0,
                                    "peak_memory": 0,
                                    "forward_cnt": 2,
                                    "pre_total_time": 3.470897674560547
                                }, {
                                    "name": "triangle_attn",
                                    "layers": [{
                                        "name": "scaled_masked_softmax",
                                        "memory": 512,
                                        "time": 465.08251536976206,
                                        "input": 516.0,
                                        "peak_memory": 542107136,
                                        "forward_cnt": 11,
                                        "pre_total_time": 5115.907669067383
                                    }],
                                    "memory": 1664,
                                    "time": 22.87912368774414,
                                    "input": 208.0,
                                    "peak_memory": 2818581504,
                                    "forward_cnt": 2,
                                    "pre_total_time": 45.75824737548828
                                }, {
                                    "name": "dense",
                                    "memory": 64,
                                    "time": 8.333802223205566,
                                    "input": 64.0,
                                    "peak_memory": 536871936,
                                    "forward_cnt": 2,
                                    "pre_total_time": 16.667604446411133
                                }],
                                "memory": 1792,
                                "time": 50.97508430480957,
                                "input": 80.0,
                                "peak_memory": 2684364288,
                                "forward_cnt": 2,
                                "pre_total_time": 101.95016860961914
                            }, {
                                "name": "post_attention_layernorm",
                                "memory": 384,
                                "time": 1.8906593322753906,
                                "input": 64.0,
                                "peak_memory": 402705408,
                                "forward_cnt": 2,
                                "pre_total_time": 3.7813186645507812
                            }, {
                                "name": "mlp",
                                "layers": [{
                                    "name": "gate_proj",
                                    "memory": 172,
                                    "time": 9.36591625213623,
                                    "input": 64.0,
                                    "peak_memory": 360711168,
                                    "forward_cnt": 2,
                                    "pre_total_time": 18.73183250427246
                                }, {
                                    "name": "up_proj",
                                    "memory": 172,
                                    "time": 8.879423141479492,
                                    "input": 64.0,
                                    "peak_memory": 360711168,
                                    "forward_cnt": 2,
                                    "pre_total_time": 17.758846282958984
                                }, {
                                    "name": "down_proj",
                                    "memory": 64,
                                    "time": 13.797521591186523,
                                    "input": 172.0,
                                    "peak_memory": 536871936,
                                    "forward_cnt": 2,
                                    "pre_total_time": 27.595043182373047
                                }],
                                "memory": 752,
                                "time": 38.39600086212158,
                                "input": 64.0,
                                "peak_memory": 1258294272,
                                "forward_cnt": 2,
                                "pre_total_time": 76.79200172424316
                            }],
                            "memory": 3312,
                            "time": 100.17907619476318,
                            "input": 64.0,
                            "peak_memory": 3942760960,
                            "forward_cnt": 2,
                            "pre_total_time": 200.35815238952637
                        }, 
                        {
                            "name": "1",
                            "layers": [{
                                "name": "input_layernorm",
                                "memory": 384,
                                "time": 2.0204544067382812,
                                "input": 64.0,
                                "peak_memory": 402705408,
                                "forward_cnt": 5,
                                "pre_total_time": 10.102272033691406
                            }, {
                                "name": "attention",
                                "layers": [{
                                    "name": "query_key_value",
                                    "memory": 192,
                                    "time": 11.822700500488281,
                                    "input": 64.0,
                                    "peak_memory": 402654208,
                                    "forward_cnt": 5,
                                    "pre_total_time": 59.113502502441406
                                }, {
                                    "name": "rotary_emb",
                                    "memory": 0,
                                    "time": 1.5523433685302734,
                                    "input": 64.0,
                                    "peak_memory": 0,
                                    "forward_cnt": 5,
                                    "pre_total_time": 7.761716842651367
                                }, {
                                    "name": "triangle_attn",
                                    "layers": [{
                                        "name": "scaled_masked_softmax",
                                        "memory": 512,
                                        "time": 2.651152403458305,
                                        "input": 516.0,
                                        "peak_memory": 542107136,
                                        "forward_cnt": 23,
                                        "pre_total_time": 60.976505279541016
                                    }],
                                    "memory": 1664,
                                    "time": 21.59562110900879,
                                    "input": 208.0,
                                    "peak_memory": 2818581504,
                                    "forward_cnt": 5,
                                    "pre_total_time": 107.97810554504395
                                }, {
                                    "name": "dense",
                                    "memory": 64,
                                    "time": 10.775327682495117,
                                    "input": 64.0,
                                    "peak_memory": 536871936,
                                    "forward_cnt": 5,
                                    "pre_total_time": 53.876638412475586
                                }],
                                "memory": 1792,
                                "time": 53.80759239196777,
                                "input": 80.0,
                                "peak_memory": 2684364288,
                                "forward_cnt": 5,
                                "pre_total_time": 269.03796195983887
                            }, {
                                "name": "post_attention_layernorm",
                                "memory": 384,
                                "time": 1.89056396484375,
                                "input": 64.0,
                                "peak_memory": 402705408,
                                "forward_cnt": 5,
                                "pre_total_time": 9.45281982421875
                            }, {
                                "name": "mlp",
                                "layers": [{
                                    "name": "gate_proj",
                                    "memory": 172,
                                    "time": 9.423637390136719,
                                    "input": 64.0,
                                    "peak_memory": 360711168,
                                    "forward_cnt": 5,
                                    "pre_total_time": 47.118186950683594
                                }, {
                                    "name": "up_proj",
                                    "memory": 172,
                                    "time": 9.195518493652344,
                                    "input": 64.0,
                                    "peak_memory": 360711168,
                                    "forward_cnt": 5,
                                    "pre_total_time": 45.97759246826172
                                }, {
                                    "name": "down_proj",
                                    "memory": 64,
                                    "time": 15.069055557250977,
                                    "input": 172.0,
                                    "peak_memory": 536871936,
                                    "forward_cnt": 5,
                                    "pre_total_time": 75.34527778625488
                                }],
                                "memory": 752,
                                "time": 39.7702693939209,
                                "input": 64.0,
                                "peak_memory": 1258294272,
                                "forward_cnt": 5,
                                "pre_total_time": 198.8513469696045
                            }],
                            "memory": 64,
                            "time": 99.24852848052979,
                            "input": 64.0,
                            "peak_memory": 851446272,
                            "forward_cnt": 2,
                            "pre_total_time": 198.49705696105957
                        }, {
                            "name": "2",
                            "layers": [{
                                "name": "input_layernorm",
                                "memory": 384,
                                "time": 2.092313766479492,
                                "input": 64.0,
                                "peak_memory": 402705408,
                                "forward_cnt": 5,
                                "pre_total_time": 10.461568832397461
                            }, {
                                "name": "attention",
                                "layers": [{
                                    "name": "query_key_value",
                                    "memory": 192,
                                    "time": 9.151220321655273,
                                    "input": 64.0,
                                    "peak_memory": 402655232,
                                    "forward_cnt": 5,
                                    "pre_total_time": 45.75610160827637
                                }, {
                                    "name": "rotary_emb",
                                    "memory": 0,
                                    "time": 1.4781951904296875,
                                    "input": 64.0,
                                    "peak_memory": 0,
                                    "forward_cnt": 5,
                                    "pre_total_time": 7.3909759521484375
                                }, {
                                    "name": "triangle_attn",
                                    "layers": [{
                                        "name": "scaled_masked_softmax",
                                        "memory": 512,
                                        "time": 2.8289504673170005,
                                        "input": 516.0,
                                        "peak_memory": 542107136,
                                        "forward_cnt": 23,
                                        "pre_total_time": 65.06586074829102
                                    }],
                                    "memory": 1664,
                                    "time": 22.263240814208984,
                                    "input": 208.0,
                                    "peak_memory": 2818582016,
                                    "forward_cnt": 5,
                                    "pre_total_time": 111.31620407104492
                                }, {
                                    "name": "dense",
                                    "memory": 64,
                                    "time": 9.025907516479492,
                                    "input": 64.0,
                                    "peak_memory": 536871936,
                                    "forward_cnt": 5,
                                    "pre_total_time": 45.12953758239746
                                }],
                                "memory": 1792,
                                "time": 50.07152557373047,
                                "input": 80.0,
                                "peak_memory": 2684364800,
                                "forward_cnt": 5,
                                "pre_total_time": 250.35762786865234
                            }, {
                                "name": "post_attention_layernorm",
                                "memory": 384,
                                "time": 1.9093513488769531,
                                "input": 64.0,
                                "peak_memory": 402705408,
                                "forward_cnt": 5,
                                "pre_total_time": 9.546756744384766
                            }, {
                                "name": "mlp",
                                "layers": [{
                                    "name": "gate_proj",
                                    "memory": 172,
                                    "time": 8.781290054321289,
                                    "input": 64.0,
                                    "peak_memory": 360711168,
                                    "forward_cnt": 5,
                                    "pre_total_time": 43.906450271606445
                                }, {
                                    "name": "up_proj",
                                    "memory": 172,
                                    "time": 9.690237045288086,
                                    "input": 64.0,
                                    "peak_memory": 360711168,
                                    "forward_cnt": 5,
                                    "pre_total_time": 48.45118522644043
                                }, {
                                    "name": "down_proj",
                                    "memory": 64,
                                    "time": 14.768743515014648,
                                    "input": 172.0,
                                    "peak_memory": 536871936,
                                    "forward_cnt": 5,
                                    "pre_total_time": 73.84371757507324
                                }],
                                "memory": 752,
                                "time": 39.36948776245117,
                                "input": 64.0,
                                "peak_memory": 1258294272,
                                "forward_cnt": 5,
                                "pre_total_time": 196.84743881225586
                            }],
                            "memory": 64,
                            "time": 98.82020950317383,
                            "input": 64.0,
                            "peak_memory": 851446272,
                            "forward_cnt": 2,
                            "pre_total_time": 197.64041900634766
                        }]
                    }],
                    "memory": 4336,
                    "time": 1621.1401224136353,
                    "input": 80.0,
                    "peak_memory": 5331085312,
                    "forward_cnt": 2,
                    "pre_total_time": 3242.2802448272705
                }],
                "memory": 4336,
                "time": 1642.3271894454956,
                "input": 16.25,
                "peak_memory": 5398523392,
                "forward_cnt": 2,
                "pre_total_time": 3284.654378890991
            }],
            "memory": 4336,
            "time": 1645.2174186706543,
            "input": 16.25,
            "peak_memory": 5398523392,
            "forward_cnt": 2,
            "pre_total_time": 3290.4348373413086
        }],
        "used_mem": 16600,
        "max_device_memory": 58960
    }
        '''
        model1 = json.loads(str1)
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        solve_graph(model1, 2, 55296 * 1024)
