# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
import shutil
import stat
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig # 导入量化配置接口

IN_MODEL_PATH = './llama2-7b' # 浮点权重输入路径
OUT_MODEL_PATH = './llama2-7b_quant' # 量化权重生成路径
NUM_LAYERS = 32 # 模型层数
ANTI_METHOD = "m1" # anti-outlier算法配置


#获取校准数据函数定义
def get_calib_dataset(_tokenizer, _calib_list):
    calib_dataset = []
    for calib_data in _calib_list:
        inputs = _tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], None, inputs.data['attention_mask']])
    return calib_dataset

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, torch_dtype=torch.float32).cpu()
print(f"loading success!")

# 准备校准数据，建议随机选择50条BoolQ数据作为校准数据
calib_list = [
    "def print_hello_world():",
    "def Fibonacci_sequence(n):",
]

#校准数据获取
dataset_calib = get_calib_dataset(tokenizer, calib_list) 

# 量化配置
# anti-outlier配置
print("anti-outlier start...")
anti_config = AntiOutlierConfig(anti_method=ANTI_METHOD)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

# 配置回退层数
print("quantization start...")
disabled_names = []
disabled_layers = [i for i in range(0, NUM_LAYERS)]
for i in disabled_layers:
    disabled_names.append(f"model.layers.{i}.mlp.down_proj")
# 配置量化参数，并返回量化配置实例
quant_config = QuantConfig(a_bit=8, w_bit=8, disable_names=disabled_names, dev_type='cpu', 
                            act_method=1, mm_tensor=False)
# 输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')

#执行量化
calibrator.run() 
print("quantization success!")

# 保存生成好的量化权重
calibrator.save(OUT_MODEL_PATH, save_type=["numpy", "safe_tensor"])
# 拷贝浮点模型配置文件
with open(os.path.join(IN_MODEL_PATH, "config.json"), 'r') as f:
    config = json.load(f)
config['quantize'] = 'w8a8'

flags = os.O_WRONLY | os.O_CREAT
modes = stat.S_IWUSR | stat.S_IRUSR
with os.fdopen(os.open(os.path.join(OUT_MODEL_PATH, "config.json"), flags, modes), "w") as f:
    json.dump(config, f)
shutil.copyfile(os.path.join(IN_MODEL_PATH, "tokenizer_config.json"), os.path.join(OUT_MODEL_PATH, "tokenizer_config.json"))
shutil.copyfile(os.path.join(IN_MODEL_PATH, "tokenizer.json"), os.path.join(OUT_MODEL_PATH, "tokenizer.json"))
shutil.copyfile(os.path.join(IN_MODEL_PATH, "tokenizer.model"), os.path.join(OUT_MODEL_PATH, "tokenizer.model"))

print(f"saved successfully")