import argparse
import json
import os
import torch.utils.data
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from transformers import AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the quantization weight of the model.")
    parser.add_argument("--float_weight", help="Path of floating-point weight.")
    parser.add_argument("--data_path", help="Path of data used for calibration.")
    parser.add_argument("--quant_weight", help="Path of quantization weight.")
    parser.add_argument("--sparse", action='store_true')
    return parser.parse_args()


def get_dataset(data_path):
    dataset = []
    with open(data_path, encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))
    return dataset


def get_calibration_dataset(tokenizer, dataset):
    calibration_dataset = []
    for data in dataset:
        text = data['inputs_pretokenized']
        inputs = tokenizer([text], return_tensors='pt').to('cpu')
        calibration_dataset.append([
            inputs.data['input_ids'],
            inputs.data['position_ids'],
            inputs.data['attention_mask']
        ])
    return calibration_dataset


def get_disable_names():
    disable_layer_indices = [0]
    disable_names = []
    for layer_index in disable_layer_indices:
        disable_names.append(f"transformer.encoder.layers.{layer_index}.self_attention.query_key_value")
        disable_names.append(f"transformer.encoder.layers.{layer_index}.self_attention.dense")
        disable_names.append(f"transformer.encoder.layers.{layer_index}.mlp.dense_h_to_4h")
        disable_names.append(f"transformer.encoder.layers.{layer_index}.mlp.dense_4h_to_h")
    disable_names.append('transformer.output_layer')
    return disable_names


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.float_weight,
        trust_remote_code=True
    ) 
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=args.float_weight,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).cpu() 

    dataset = get_dataset(args.data_path)
    calibration_dataset = get_calibration_dataset(tokenizer, dataset)
    disable_names = get_disable_names()
    
    if args.sparse:
        sparse_config = QuantConfig(
            w_bit=4,
            disable_names=disable_names, # 不进行量化的权重名
            dev_type='cpu',
            act_method=3,
            pr=2.0,
            fraction=0.011,
            nonuniform=False,
            mm_tensor=False,
            co_sparse=True # 使用稀疏配置
        )
        calibrator = Calibrator(model, sparse_config, calib_data=calibration_dataset, disable_level="L0")
        calibrator.run(int_infer=False)
    else:
        quant_config = QuantConfig(
            w_bit=8,
            disable_names=disable_names,
            dev_type='cpu',
            act_method=3,
            pr=1.0,  # pr=1.0 可以关闭导出的量化权重的随机性
            mm_tensor=False,
            w_hessian=False
        )
        calibrator = Calibrator(model, quant_config, calib_data=calibration_dataset, disable_level="L0")
        calibrator.run()
    calibrator.save(args.quant_weight)

