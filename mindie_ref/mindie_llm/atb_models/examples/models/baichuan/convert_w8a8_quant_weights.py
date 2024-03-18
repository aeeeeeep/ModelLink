import os
import json
import torch
import torch_npu # npu进行量化
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

input_fp16_path = 'your model path'
output_w8a8_path = 'your output path'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=input_fp16_path, trust_remote_code=True) 
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=input_fp16_path, trust_remote_code=True).float().cpu()
# model = model.half().npu() # 如果需要使用npu进行量化
# 获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list, device="cpu"):  # device="npu:0" 如果需要使用npu进行量化
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['attention_mask'].to(device)
            ])
    return calib_dataset
calib_set = [
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n编写中小学教科书的直接依据是____。\nA. 《中华人民共和国教育法》\nB. 课程计划\nC. 课程标准\nD. 课程表\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n下列关于课程的三种文本表现形式说法正确的是____\nA. 课程计划是由当地教育主管部门制订的\nB. 课程标准是依据课程计划制定的\nC. 课程标准的核心是实施建议\nD. 教材编写的基本方式有直线式、螺旋式、交叉式\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n悦悦是一名右耳失聪的残疾儿童，活动课上有时会听不清楚周老师所讲的内容，因此经常提问题。对此，周老师应当采取的措施是____。\nA. 给予悦悦更多的帮助和指导\nB. 指导家长带悦悦回家自学\nC. 建议家长将悦悦转到特殊幼儿园\nD. 照顾大多数幼儿，不理会悦悦\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n内流河也称“内陆河”，是指没有流入海洋的河流，大多分布在大陆内部干燥地区，上游降水或冰雪融水为其主要补给水源，最终消失于沙漠或注入内陆湖泊。下列中国内流河中，最长的是____。\nA. 塔里木河\nB. 柴达木河\nC. 尼雅河\nD. 疏勒河\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n学校规定学生不能烫染头发，但是小文为了彰显个性，在假期把头发染成了棕色。面对小文的情况，教师应该怎样处理？____\nA. 年轻人追求个性是合情合理的，应该宽容对待\nB. 违反学校的校规，应该严格处分\nC. 强制要求小文将头发颜色染回来才可以进校门\nD. 探明小文违反校规的原因，并对其进行劝导和教育\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n张老师根据自己班级的情况，为解决班级内部班干部的人际关系问题，建立和谐融洽的班级氛围，自主开发了“和谐人际”的班级课程，这体现了教师____。\nA. 是教育教学的研究者\nB. 是课程的建设者和开发者\nC. 是学生学习的促进者\nD. 是社区型的开放教师\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n刘老师工作很负责，学生在学校出现一点问题他就会与家长联系，在与家长沟通时他经常以前辈的姿态对待家长，对家长的教育方式指指点点。刘老师的做法____。\nA. 正确，老师就应该与家长经常沟通\nB. 正确，老师的经验比家长丰富，应该多指导家长\nC. 不正确，教师没有权利指导家长\nD. 不正确，教师应该与家长建立平等的沟通关系，尊重家长的人格\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n在古代印度，有一户人家经营一家棉布店销售自己手工制作的衣服。你认为这户人家属于哪个等级？____\nA. 婆罗门\nB. 刹帝利\nC. 吠舍\nD. 首陀罗\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n“小型分散，便于开展多种多样的活动，满足学生不同的兴趣、爱好，发展学生的才能，使学生得到更多的学习和锻炼的机会。”这种课外活动的形式是____。\nA. 科技活动\nB. 学科活动\nC. 个人活动\nD. 小组活动\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n小红每天晚上临睡前都要多次反复检查自己的书包，确保带齐了第二天需要用的教材和文具。她明知道没有这个必要，但就是控制不住。她可能出现了____。\nA. 抑郁症\nB. 焦虑症\nC. 强迫症\nD. 恐惧症\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n国家管理和评价课程的基础是____。\nA. 课程计划\nB. 课程标准\nC. 教学目标\nD. 教育目的\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n儿童坚持性发生明显质变的年龄约在____\nA. 3～4岁\nB. 4～5岁\nC. 5～6岁\nD. 6岁以后\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n《红楼梦》中人物众多、关系繁杂。为了帮助读者阅读，许多红学爱好者都在网络上发布了自己整理制作的主要人物关系图。这属于____。\nA. 纲要策略\nB. 精细加工策略\nC. 资源管理策略\nD. 监控策略\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n学期结束时，班主任王老师会对学生思想品德的发展变化情况进行评价。这项工作属于____。\nA. 工作总结\nB. 工作计划\nC. 操行评定\nD. 建立学生档案\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n人们常说：“教学有法而教无定法。”这反映了教师的劳动具有____。\nA. 连续性\nB. 示范性\nC. 长期性\nD. 创造性\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n县级以上地方各级人民代表大会是县级以上地方国家权力机关，其职权不包括____。\nA. 改变或撤销本级人大常务委员会不适当的决定\nB. 选举并有权罢免本级人民法院院长\nC. 批准本行政区域内的预算执行情况的报告\nD. 决定并宣布下一级行政区城进入紧急状态\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n在心理健康课上，同一批学生在第二次进行同样内容的人格测验时获得的分数与上次测验差别较大。这说明该测验存在的问题是____。\nA. 信度问题\nB. 效度问题\nC. 难度问题\nD. 区分度问题\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n李老师在教学生区分形近字“渴”“竭”“碣”“谒”时，将四个字相同的右半部分用白色粉笔写出，相异的左半部分用彩色粉笔写出。李老师运用了知觉的____。\nA. 整体性\nB. 选择性\nC. 理解性\nD. 恒常性\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n兰兰学会走路后,就要很喜欢尝试自己穿衣、吃饭、捡东西,喜欢探索周围世界。按照埃里克森人格发展阶段理论,兰兰所处的发展阶段是____\nA. 信任对怀疑\nB. 自立对羞怯\nC. 主动感对内疚感\nD. 勤奋感对自卑感\nAnswer:",
  "The following are multiple choice questions (with answers) about  teacher qualification.\n\n下列对于多动症的说法，不正确的是____\nA. 由多种原因引起的一组综合征\nB. 某种神经递质的缺陷可诱发该病\nC. 神经髓鞘发育落后可诱发该病\nD. 营养不良可诱发该病\nAnswer: D\n\n学习迁移发生的必要条件是两种学习活动之间存在共同原理，学习迁移产生的关键是学习者通过活动能概括出其共同原理。持这种观点的迁移理论被称为____\nA. 形式训练说\nB. 相同要素说\nC. 概括化理论\nD. 关系理论\nAnswer: C\n\nExcel中，通常在单元格内出现“####”符号时，表明____。\nA. 显示的是字符串“####”\nB. 列宽不够，无法显示数值数据\nC. 数值溢出\nD. 计算错误\nAnswer: B\n\n第二次世界大战开始时间是____。\nA. 1914年\nB. 1918年\nC. 1939年\nD. 1945年\nAnswer: C\n\n在日常生活中，我们经常会接触一些民谚、俗语，这些民谚、俗语蕴含着丰富的物理知识。下列民谚、俗语蕴含的物理知识所属领域不同的是____。\nA. 坐井观天，所见甚少\nB. 瑞雪兆丰年\nC. 酒香不怕巷子深\nD. 下雪不寒化雪寒\nAnswer: A\n\n杨老师在教授生字词的过程中发现部分学生有缺笔少画的现象，于是他把“小学生缺笔少画现象的原因及对策研究”作为研究课题，拟订相应的研究计划，在工作中收集、整理相关资料并实施教学措施，最后根据反馈信息调整教学方案。这种研究方法属于____。\nA. 教育行动研究法\nB. 教育实验法\nC. 教育叙事研究法\nD. 个案研究法\nAnswer:"
]
dataset_calib = get_calib_dataset(tokenizer, calib_set)
"""
对于linear算子中的激活值如果有表示范围过大，或者“尖刺”的异常值过多，
需要使用anti outleir功能，使用方法如下
"""
anti_config = AntiOutlierConfig(anti_method="m2", dev_type="cpu")  # dev_type="npu", dev_id=0  如果需要使用npu进行量化
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config, norm_class_name="RMSNorm")
anti_outlier.process()
"""
下面是回退层的设置，因为w8a8的对激活值也进行了量化，会有部分网络层对激活值的表示
范围较为敏感所以需要回退这些网络层使用浮点权重进行计算。
"""

disable_names=[]
llama_layers = 32
disable_idx_lst = list(range(llama_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)
# w_sym=True：对称量化，w_sym=False：非对称量化;
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8, 
    disable_names=disable_names, 
    dev_type='cpu',  # dev_type="npu", dev_id=0  如果需要使用npu进行量化
    act_method=3, 
    pr=1.0, 
    w_sym=True, 
    mm_tensor=False，
    disable_latest_linear=False
)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()  # 执行PTQ量化校准
calibrator.save(output_w8a8_path, save_type=["safe_tensor"])  #"safe_tensor"对应safetensors格式权重，"numpy"对应npy格式权重

