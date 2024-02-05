# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import argparse
import torch
import torch_npu
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

SEQ_LEN_OUT = 32

'''
# for 中文模型
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
'''

calib_list = ["Where is the capital of China?",
              "Please make a poem:",
              "I want to learn python, how should I learn it?",
              "Please help me write a job report on large model inference optimization:",
              "What are the most worth visiting scenic spots in China?"]


def get_calib_dataset(tokenizer_in, data):
    calib_dataset = []
    for calib_data in data:
        inputs = tokenizer_in([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="quantizing model weights")
    parser.add_argument(
        "--input_path",
        default="/data/models/llama2-7b",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default="/data/models/llama2-7b_quant",
        help="Location to write the weights",
    )
    parser.add_argument(
        "--disable_level",
        default='L8',
        help="number of layers that don't need to be quantized",
    )

    args = parser.parse_args()
    # load float model weight
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.input_path,
                                            trust_remote_code=True)
    print("load tokenizer success!")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.input_path,
                                                torch_dtype=torch.float32, trust_remote_code=True).cpu()
    print("load model success!")

    # quantization
    dataset_calib = get_calib_dataset(tokenizer, calib_list)

    quant_config = QuantConfig(w_bit=8, disable_names=[], dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level=args.disable_level) # disable_level: L1-L5，量化权重回退1-5层
    calibrator.run()

    # save quant weight
    calibrator.save(args.output_path) 
    print('Save quant weight success!')
