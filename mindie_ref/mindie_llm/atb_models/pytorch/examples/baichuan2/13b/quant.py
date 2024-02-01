# coding: utf-8

import tempfile
import os
import time

import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# from anti_outlier.anti_outlier import AntiOutlier
# from anti_outlier.config import AntiOutlierConfig

"""
["费用报销需要提供哪些材料？"],
["微信支付可以支持哪些银行卡？"],
["简历中应该如何突出重点？"],
["海外留学需要注意哪些事项？"],
["云计算对于企业有哪些好处？"],
["常见的投资方式有哪些？"],
["什么是股票的基本面分析？"],
["运动员如何保持良好的竞技状态？"],
["暴雨天气应该注意哪些安全事项？"],
["驾照考试一共有几个科目？"],
["食品安全检测的流程是什么？"],
["股票交易中的龙头股是什么？"],
["网络攻击有哪些形式？"],
["新能源汽车的优势是什么？"],
["What are the benefits of cloud computing for businesses?"],
["What documents are required for expense reimbursement?"],
["How to highlight key points in a resume?"],
["What should be paid attention to when studying abroad?"],
["Which banks does WeChat payment support?"],
["What are the common investment methods?"],
["What is the process of food safety inspection?"],
["What is the basic analysis of stock fundamentals?"],
["How do athletes maintain good athletic performance?"],
["What safety precautions should be taken in rainy weather?"],
["What are the subjects of the driver's license exam?"],
["What are the types of cyber attacks?"],
["What is the concept of leading stocks in stock trading?"],
["What should be noted in the use of electronic invoices?"],
["What are the advantages of new energy vehicles?"],
["如何有效管理个人财务？"],
["什么是人工智能的发展趋势？"],
["如何设计一个用户友好的网站界面？"],
["为什么要进行环境保护？"],
["如何预防常见的网络安全漏洞？"],
["如何培养良好的沟通能力？"],
["学习一门外语需要多长时间？"],
["什么是健康的饮食习惯？"],
["什么是心理健康？如何保持心理健康？"],
["如何应对工作压力？"],
["How to effectively manage personal finances?"],
["What are the development trends of artificial intelligence?"],
["How to design a user-friendly website interface?"],
["Why is environmental protection important?"],
["How to prevent common network security vulnerabilities?"],
["How to cultivate good communication skills?"],
["How long does it take to learn a foreign language?"],
["What are healthy eating habits?"],
["What is mental health and how to maintain it?"],
"""


calib_list = [
    ["电子发票有哪些注意事项？"],
    ["费用报销需要提供哪些材料？"],
    ["微信支付可以支持哪些银行卡？"],
    ["简历中应该如何突出重点？"],
    ["海外留学需要注意哪些事项？"],
    ["云计算对于企业有哪些好处？"],
    ["常见的投资方式有哪些？"],
    ["什么是股票的基本面分析？"],
    ["运动员如何保持良好的竞技状态？"],
    ["暴雨天气应该注意哪些安全事项？"],
    ["驾照考试一共有几个科目？"],
    ["食品安全检测的流程是什么？"],
    ["股票交易中的龙头股是什么？"],
    ["网络攻击有哪些形式？"],
    ["新能源汽车的优势是什么？"],
    ["What are the benefits of cloud computing for businesses?"],
    ["What documents are required for expense reimbursement?"],
    ["How to highlight key points in a resume?"],
    ["What should be paid attention to when studying abroad?"],
    ["Which banks does WeChat payment support?"],
    ["What are the common investment methods?"],
    ["What is the process of food safety inspection?"],
    ["What is the basic analysis of stock fundamentals?"],
    ["How do athletes maintain good athletic performance?"],
    ["What safety precautions should be taken in rainy weather?"],
    ["What are the subjects of the driver's license exam?"],
    ["What are the types of cyber attacks?"],
    ["What is the concept of leading stocks in stock trading?"],
    ["What should be noted in the use of electronic invoices?"],
    ["What are the advantages of new energy vehicles?"],
    ["如何有效管理个人财务？"],
    ["什么是人工智能的发展趋势？"],
    ["如何设计一个用户友好的网站界面？"],
    ["为什么要进行环境保护？"],
    ["如何预防常见的网络安全漏洞？"],
    ["如何培养良好的沟通能力？"],
    ["学习一门外语需要多长时间？"],
    ["什么是健康的饮食习惯？"],
    ["什么是心理健康？如何保持心理健康？"],
    ["如何应对工作压力？"],
    ["How to effectively manage personal finances?"],
    ["What are the development trends of artificial intelligence?"],
    ["How to design a user-friendly website interface?"],
    ["Why is environmental protection important?"],
    ["How to prevent common network security vulnerabilities?"],
    ["How to cultivate good communication skills?"],
    ["How long does it take to learn a foreign language?"],
    ["What are healthy eating habits?"],
    ["What is mental health and how to maintain it?"],
    ["How to cope with work-related stress?"]
]
"""
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='/code/models/baichuan2/baichuan2_cmh',
                                          trust_remote_code=True, use_fast=False, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='/code/models/baichuan2/baichuan2_cmh',
                                             torch_dtype=torch.float32, trust_remote_code=True).cpu()
"""


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='/code/models/baichuan2/baichuan2_cmh/13b/',
                                          trust_remote_code=True, use_fast=False, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='/code/models/baichuan2/baichuan2_cmh/13b/',
                                             torch_dtype=torch.float32, trust_remote_code=True).cpu()

print(model)
for name in model.state_dict():
    print(name)
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

"""
at_config = AntiOutlierConfig(w_bit=8,
                              a_bit=8,
                              # w_signed=True,
                              # a_signed=False,
                              # w_sym=True,
                              # a_sym=False,
                              # alpha=0.5,
                              # os_k=100,
                              # ch_align=True,
                              # w_adjust=True,
                              anti_method="m2",
                              dev_type="cpu"
                              )
                              
model_type = "Llama"
anti_outlier = AntiOutlier(model, dataset_calib, cfg=at_config, model_type=model_type)

processed_model_outpath = "/data1/c30054301/baichuan2-13b/anti-outlier/" # 异常值抑制后的模型保存路径

print(">>>>anti_outlier process starts >>>>")
anti_outlier.process()
print(anti_outlier.model)
model = anti_outlier.model
del anti_outlier

tokenizer.save_pretrained(processed_model_outpath)
model.save_pretrained(processed_model_outpath)

torch.cuda.empty_cache()

print("<<<< anti_outlier process finished, model outputed <<<<")
"""

# disable_names = ["lm_head"]
disable_names = []
layer_num = 10
# for layer_index in range(layer_num):
# w_pack_name = "model.layers.{}.self_attn.W_pack".format(layer_index)
# o_proj_name = "model.layers.{}.self_attn.o_proj".format(layer_index)
# up_proj_name = "model.layers.{}.mlp.up_proj".format(layer_index)
# gate_proj_name = "model.layers.{}.mlp.gate_proj".format(layer_index)
# down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
# disable_names.append(w_pack_name)
# disable_names.append(o_proj_name)
# disable_names.append(up_proj_name)
# disable_names.append(gate_proj_name)
# disable_names.append(down_proj_name)
# for layer_index in range(layer_num):
# w_pack_name = "model.layers.{}.self_attn.W_pack".format(layer_index + 30)
# o_proj_name = "model.layers.{}.self_attn.o_proj".format(layer_index + 30)
# up_proj_name = "model.layers.{}.mlp.up_proj".format(layer_index + 30)
# gate_proj_name = "model.layers.{}.mlp.gate_proj".format(layer_index + 30)
# down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index + 30)
# disable_names.append(w_pack_name)
# disable_names.append(o_proj_name)
# disable_names.append(up_proj_name)
# disable_names.append(gate_proj_name)
# disable_names.append(down_proj_name)
# disable_names.append("lm_head.weight")

#disable_idx_lst = [0,1,2,3,4,7,9,10,17,18,19,20,22,23,24,26,27,36,37,38]
#disable_idx_lst = [0,1,2,3,4,7,9,10,36,37,38,39]
# disable_idx_lst = [3,4,9,10,17,18,20,23,24,36,38,39]
# disable_idx_lst = [10,36,4,23,20,3,17,18,24,26,9,22,2,37,19,1,0]

disable_idx_lst = [0,1,2,3,4,7,9,10,17,18,19,20,22,23,24,26,36,37,38,39]
for layer_index in disable_idx_lst:
    w_pack_name = "model.layers.{}.self_attn.W_pack".format(layer_index)
    o_proj_name = "model.layers.{}.self_attn.o_proj".format(layer_index)
    up_proj_name = "model.layers.{}.mlp.up_proj".format(layer_index)
    gate_proj_name = "model.layers.{}.mlp.gate_proj".format(layer_index)
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(w_pack_name)
    disable_names.append(o_proj_name)
    disable_names.append(up_proj_name)
    disable_names.append(gate_proj_name)
    disable_names.append(down_proj_name)
#disable_names.append("lm_head.weight")
disable_names.append("lm_head")
input_shape = None
# keep_acc = {'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}
quant_config = QuantConfig(
    disable_names=disable_names, dev_type='cpu', act_method=3, mm_tensor=False, w_hessian=False, pr=1.0
)
disable_level="L0"
# calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level=disable_level)

quanted_weight_outpath = "/home/kbx/baichuan/data_1212_pr1_ctl"

print(">>>> ptq process starts >>>>")
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level=disable_level)
calibrator.run(calib_amp=10)
calibrator.save(quanted_weight_outpath)      #使用save()保存模型量化参数，请根据实际情况修改路径
print('<<<< ptq process finished, quanted weights get stored <<<<')

