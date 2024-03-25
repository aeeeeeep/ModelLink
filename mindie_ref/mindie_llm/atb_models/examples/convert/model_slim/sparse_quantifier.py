import argparse
from enum import Enum

import json
import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator as SparseQuantCalibrator
from modelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig as SparseQuantConfig
from ..convert_utils import copy_tokenizer_files, modify_config


class ActMethod(Enum):
    DATA_FREE = 0.0
    LABEL_FREE_MINMAX = 1.0
    LABEL_FREE_HISTOGRAM = 2.0
    LABEL_FREE_AUTO = 3.0


class SparseQuantifier:
    def __init__(self, model_path_or_name, quant_config=None, device_type='cpu'):
        self.device_type = device_type
        self.dtype = torch.float16 if self.device_type == 'npu' else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path_or_name,
            low_cpu_mem_usage=True, torch_dtype=self.dtype)
        self.model.to(self.device_type)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_name, use_fast=False, trust_remote_code=True, legacy=False
        )

        if quant_config is not None:
            self.quant_config = quant_config
        else:
            self.quant_config = self.update_quant_config()

    def get_tokenized_data(self, input_texts,
                           input_ids_name='input_ids',
                           attention_mask_name='attention_mask'):
        tokenized_data = []
        for text in input_texts:
            inputs = self.tokenizer([text], return_tensors='pt').to(self.device_type)
            position_ids = torch.zeros_like(inputs.data[input_ids_name], dtype=torch.long)
            tokenized_data.append(
                [inputs.data[input_ids_name], position_ids, inputs.data[attention_mask_name]])
        return tokenized_data

    def update_quant_config(self,
                            act_method=1,
                            disable_names=None,
                            w_bit=4,
                            fraction=0.011):
        self.quant_config = SparseQuantConfig(w_bit=w_bit,
                                              disable_names=disable_names,
                                              dev_type=self.device_type,
                                              act_method=act_method,  #
                                              pr=1.0,  # randseed
                                              fraction=fraction,
                                              nonuniform=False,
                                              mm_tensor=False,
                                              co_sparse=True)
        return self.quant_config

    def sparse_convert(self, input_texts, save_path, input_ids_name='input_ids', attention_mask_name='attention_mask'):
        tokenized_data = self.get_tokenized_data(input_texts, input_ids_name, attention_mask_name)
        calibrator = SparseQuantCalibrator(self.model, self.quant_config, calib_data=tokenized_data,
                                           disable_level='L0')  # 内部回退两层
        calibrator.run(int_infer=False)
        calibrator.save(save_path, save_type=['safe_tensor'])


def load_jsonl(dataset_path, key_name='inputs_pretokenized'):
    dataset = []
    with open(dataset_path, encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            text = data[key_name]
            dataset.append(text)
    return dataset[:1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help="model and tokenizer path",
                        default='/home/data/acltransformer_testdata/weights/llama2/llama-2-7b',
                        )
    parser.add_argument('--save_directory',
                        default='/home/data/acltransformer_testdata/weights/model_slim/sparse_quant_7b_step1',
                        )
    parser.add_argument(
        '--calib_texts',
        type=str,
        nargs='+',
        default=["What's deep learning?"])
    parser.add_argument(
        '--calib_file',
        type=str,
        help='CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=f"{os.path.join(os.path.dirname(__file__), 'teacher_qualification.jsonl')}")
    parser.add_argument('--w_bit', type=int, default=4)
    parser.add_argument('--disable_names', type=str, nargs='+', default=None)
    parser.add_argument('--device_type', type=str, default='cpu')
    parser.add_argument('--fraction', type=float, default=0.011)
    parser.add_argument("--act_method", type=int, choices=[1, 2, 3, 4], default=1,
                        help=" `0`: `Data-Free`, `1`: `Label-Free`, `2`: `Label-Free-Histogram`, `3`: `Label-Free-Auto`")
    parser.add_argument('--input_ids_name', type=str, default='input_ids')
    parser.add_argument('--attention_mask_name', type=str, default='attention_mask')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    calib_file = args.calib_file
    calib_texts = load_jsonl(calib_file) if calib_file else args.calib_texts
    model_path = args.model_path
    save_directory = args.save_directory

    quant_conf = SparseQuantConfig(w_bit=args.w_bit,
                                   disable_names=args.disable_names,
                                   dev_type=args.device_type,
                                   act_method=args.act_method,
                                   fraction=args.fraction,
                                   pr=1.0,  # randseed
                                   nonuniform=False,
                                   mm_tensor=False,
                                   co_sparse=True)

    sparse_quantifier = SparseQuantifier(model_path, quant_conf)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    sparse_quantifier.sparse_convert(calib_texts, save_directory, args.input_ids_name, args.attention_mask_name)
    modify_config(model_path, save_directory, torch.float16, 'w8a8sc')
    copy_tokenizer_files(model_path, save_directory)
