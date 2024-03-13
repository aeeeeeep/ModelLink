import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights for CodeGeex2-6B")
    parser.add_argument("--model_path", type=str, required=True, help="The path to model float weights")
    parser.add_argument("--save_path", type=str, default="./quant_weight_glm", help="The path to save quant weights")
    parser.add_argument("--dataset_path", type=str, required=True, help="The dataset path")

    return parser.parse_args()


# 获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list, device="cpu"):  # device="npu:0" 如果需要使用npu进行量化
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['position_ids'].to(device),
            inputs.data['attention_mask'].to(device)
            ])
    return calib_dataset


disable_names = ['transformer.encoder.layers.0.self_attention.query_key_value',
'transformer.encoder.layers.0.mlp.dense_4h_to_h',
'transformer.encoder.layers.1.self_attention.query_key_value',
'transformer.encoder.layers.1.mlp.dense_h_to_4h',
'transformer.encoder.layers.1.mlp.dense_4h_to_h',
'transformer.encoder.layers.2.self_attention.query_key_value',
'transformer.encoder.layers.2.mlp.dense_h_to_4h',
'transformer.encoder.layers.2.mlp.dense_4h_to_h',
'transformer.encoder.layers.3.self_attention.query_key_value',
'transformer.encoder.layers.4.self_attention.query_key_value',
'transformer.encoder.layers.5.self_attention.query_key_value',
'transformer.encoder.layers.6.self_attention.query_key_value',
'transformer.encoder.layers.7.self_attention.query_key_value',
'transformer.encoder.layers.8.self_attention.query_key_value',
'transformer.encoder.layers.9.self_attention.query_key_value',
'transformer.encoder.layers.11.self_attention.query_key_value',
'transformer.encoder.layers.17.mlp.dense_4h_to_h',
'transformer.encoder.layers.23.mlp.dense_4h_to_h',
'transformer.encoder.layers.27.mlp.dense_4h_to_h',
'transformer.output_layer']

quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8, 
    disable_names=disable_names, 
    dev_type='cpu',  # dev_type="npu", dev_id=0  如果需要使用npu进行量化
    act_method=3, 
    pr=1.0, 
    w_sym=True, 
    mm_tensor=False
)


def main():
    args = parse_args()
    fp16_path = args.model_path  # 原始浮点模型路径
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=fp16_path, trust_remote_code=True) 
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=fp16_path, trust_remote_code=True).float().cpu()

    calib_set = []
    with open(args.dataset_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            calib_set.append(line)

    dataset_calib = get_calib_dataset(tokenizer, calib_set)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(args.save_path, save_type=["safe_tensor", "numpy"])  # "safe_tensor"对应safetensors格式权重，"numpy"对应npy格式权重

if __name__ == '__main__':
    main()