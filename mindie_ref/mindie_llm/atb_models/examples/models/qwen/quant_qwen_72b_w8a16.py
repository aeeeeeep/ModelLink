import os
import sys
import json
import inspect
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def load_tokenizer_and_model(fp16_path):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=fp16_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True
    ) 
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=fp16_path,
        torch_dtype=torch.float32, trust_remote_code=True
    ).cpu()
    return tokenizer, model


def infer(tokenizer, model, query, model_params=None):
    """
    推理代码
    :param query:
    :param model_params:
    :return:
    """
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    with torch.no_grad():
        start_time = time.time()
        model_params = model_params if model_params is not None else {}
        pred = model.generate(**inputs, **model_params)
        end_time = time.time()
        time_cost = end_time - start_time
    output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    print(output)
    print(f"cost {time_cost}s")
    new_tokens = len(pred[0]) - len(inputs.input_ids[0])
    print(f"generate {new_tokens} new tokens, ({new_tokens / time_cost:.2f} tokens/s)")
    return output


def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to("cpu")
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset


def main(fp16_path, quant_save_path):    
    tokenizer, model = load_tokenizer_and_model(fp16_path)
    print(f">>>> Load model from {os.path.basename(inspect.getmodule(model).__file__)} successfully.")

    print(">>>> Infer before calibrating")
    infer(
        tokenizer,
        model,
        "登鹳雀楼->王之涣\n夜雨寄北->",
        {"max_new_tokens": 32, "do_sample": False, "repetition_penalty": 1.1}
    )
    infer(
        tokenizer,
        model,
        "Hamlet->Shakespeare\nOne Hundred Years of Solitude->",
        {"max_new_tokens": 32, "do_sample": False, "repetition_penalty": 1.1}
    )

    data_list = [
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
    dataset_calib = get_calib_dataset(tokenizer, data_list)
    print(">>>> Calibrator dataset is ready.")

    disable_names = ['lm_head']
    print(f">>>> Disable layers: {disable_names}")

    quant_config = QuantConfig(
        w_bit=8,  # 权重量化位数
        a_bit=16,  # 激活值量化位数
        disable_names=disable_names,  # 不做量化的层（通常是空list）
        dev_type='cpu',
        act_method=3,  # 激活量化方法，建议方法3（1：min-max；2：histogram；3：自动混合量化）
        pr=1.0,  # 量化正则百分比，建议0.5
        w_sym=False,  # 对称/非对称量化，True为对称量化，False为非对称量化
        mm_tensor=False  # 权重量化粒度，True为per-tensor量化，False为per-channel量化（大模型场景建议False）
    )
    
    print(">>>> Constructing calibrator...")
    calibrator = Calibrator(
        model,
        quant_config,
        calib_data=dataset_calib,
        disable_level='L0'  # 自动回退等级，根据精度损失程度增加不量化的层（L0~L5，L0为不回退，精度损失明显时可适当提升等级）
    )
    
    print(">>>> Calibrating...")
    calibrator.run()  # 执行PTQ量化校准

    calibrator.save(quant_save_path, save_type=["safe_tensor"])
    print(f">>>> Quant weights saved in {quant_save_path}.")

    print(">>>> Infer after calibrating.")
    infer(
        tokenizer,
        model,
        "登鹳雀楼->王之涣\n夜雨寄北->",
        {"max_new_tokens": 32, "do_sample": False, "repetition_penalty": 1.1}
    )
    infer(
        tokenizer,
        model,
        "Hamlet->Shakespeare\nOne Hundred Years of Solitude->",
        {"max_new_tokens": 32, "do_sample": False, "repetition_penalty": 1.1}
    )


if __name__ == "__main__":
    fp16_path = sys.argv[1]
    quant_save_path = sys.argv[2]
    main(fp16_path, quant_save_path)
