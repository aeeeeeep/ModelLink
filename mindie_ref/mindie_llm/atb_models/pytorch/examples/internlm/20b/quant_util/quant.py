import torch
import torch.utils.data
import time
import argparse
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # 导入量化配置接口
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from configuration_internlm import InternLMConfig


def infer(tokenizer, model, query):
    """
        推理代码

        :param query:
        :return:
    """
    batch_input = [query]
    seq_len_in, seq_len_out = 128, 32
    inputs = tokenizer(batch_input, return_tensors="pt", padding="max_length", max_length=seq_len_in)
    with torch.no_grad():
        start_time = time.time()
        output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=seq_len_out)
        end_time = time.time()
        time_cost = end_time - start_time
        res = tokenizer.decode(output[0], skip_special_tokens=True)
        print(res)
        print(f"cost {time_cost}s")
        new_tokens = len(output[0]) - len(inputs.input_ids[0])
        print(f"generate {new_tokens} new tokens，({new_tokens / time_cost:.2f} tokens/s")
        return output


# 获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset


def main(args):
    # for local path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_path,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=args.model_path,
                                      torch_dtype=torch.float32, trust_remote_code=True).cpu()

    # 准备校准数据，请根据实际情况修改

    calib_list = [
        ["中国的首都在哪里？"],
        ["请做一首诗歌："],
        ["我想要学习python，该怎么学习？"],
        ["请帮我写一篇关于大模型推理优化的任职报告："],
        ["中国最值得去的几个景点"],
        ["在原始社会，主要承担教师职责的是："],
        ["下列关于税法基本原则的表述中，不正确的是:"],
        ["关于战略管理表述错误的是"],
        ["对于下列长期股权投资，应采用权益法核算的是:"],
        ["在波长为λ的驻波中两个相邻波节之间的距离为："],
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

    # 量化配置
    dataset_calib = get_calib_dataset(tokenizer, calib_list)  # 校准数据获取

    # 使用QuantConfig接口，配置量化参数，并返回量化配置实例
    disable_names = ['transformer.output_layer']
    quant_config = QuantConfig(w_bit=8, disable_names=disable_names, dev_type='cpu', act_method=3, pr=0.5,
                               mm_tensor=False, w_hessian=False)

    # 使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
    anti_config = AntiOutlierConfig(anti_method=args.anti_method)
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config, model_type='Llama')
    anti_outlier.process()
    tokenizer.save_pretrained(os.path.join(args.antioutlier_output_path))
    model.save_pretrained(args.antioutlier_output_path)
    torch.save(model, os.path.join(args.antioutlier_output_path, args.model_name + '_' + args.anti_method + '.pth'))
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L5')
    calibrator.run()  # 使用run()执行量化
    print(model)
    calibrator.save(args.quant_output_path)  # 使用save()保存模型量化参数，请根据实际情况修改路径
    print('Save quant weight success!')
    print('伪量化中，可退出')
    if args.print_example:
        infer(tokenizer=tokenizer, model=model, query='登鹳雀楼->王之涣\n夜雨寄北->')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model Quant")
    parser.add_argument(
        "--model_name",
        default='internLM_20B',
        help="模型名称",
    )
    parser.add_argument(
        "--model_path",
        default='./',
        help="原始模型路径",
    )
    parser.add_argument(
        "--quant_output_path",
        default='./quant/',
        help="量化输出路径",
    )
    parser.add_argument(
        "--antioutlier_output_path",
        default='./antioutlier/',
        help="离群点抑制输出路径",
    )
    parser.add_argument(
        "--print_example",
        default=False,
        help="输出伪量化结果",
    )
    parser.add_argument(
        "--anti_method",
        default="m2",
        help="离群点抑制算法类型",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
