import os
import torch
import numpy as np

weight_path = '/data/models/internlm-20b/quant'
input_scale_dict = np.load(os.path.join(weight_path, "input_scale.npy"), allow_pickle=True).item()

keys = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.gate_proj', 'mlp.up_proj', 'self_attn.o_proj', 'mlp.down_proj']
layer_num = 60

name = "model.layers.{}.{}"
for i in range(layer_num):
    for key in keys:
        key_name = name.format(i, key)
        # print(f'key_name = {key_name}')
        if key_name not in input_scale_dict:
            print(f'第{i}层为回退层')