# CodeLlama-34b 推理指导

## 概述

CodeLlama是一组经过预训练和微调的生成文本模型，其参数量从70亿到340亿不等。

模型下载：https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf

### 模型功能:

- [√] Code completion.
- [ ] Infilling.
- [√] Instructions / chat.
- [ ] Python specialist.

### 输入输出

输入：仅对文本输入进行建模
输出：仅生成文本

### 状态

这是在离线数据集上训练的静态模型。Code Llama - Instruct 的未来版本将发布，因为我们根据社区反馈改进了模型安全性。

### 预期用途

#### 预期用例

CodeLlama及其变体旨在用于英语和相关编程语言的商业和研究用途；
基本模型CodeLlama可以适应各种代码合成和理解任务，CodeLlama - Python专门设计用于处理Python编程语言，CodeLlama - Instruct旨在更安全地用于代码助手和生成应用程序。

#### 超出范围的使用

以任何违反适用法律或法规（包括贸易合规法）的方式使用；
以英语以外的语言使用；
以CodeLlama及其变体的可接受使用政策和许可协议禁止的任何其他方式使用。

### 道德考虑和限制

CodeLlama及其变体是一种新技术，使用时存在风险。迄今为止进行的测试是用英语进行的，没有涵盖，也不能涵盖所有情况。由于这些原因，与所有LLM一样，CodeLlama的潜在输出无法提前预测，并且在某些情况下，该模型可能会对用户提示产生不准确或令人反感的响应。因此，在部署CodeLlama的任何应用程序之前，开发人员应执行针对其模型的特定应用程序量身定制的安全测试和调整。

## 加速库接入

目前加速库代码已归一至llama
> 工作目录和llama一致,  `cd ./pytorch/examples/llama`

## 模型推理
1. 如果跑多卡多芯推理，需要先切分模型权重，切分方法如下：

- 修改代码

  1. 修改`cut_weight.sh`中`input_dir`为实际存放模型权重的路径
  
  2. 修改`cut_weight.sh`中`output_dir`为自定义路径，用于存放切分后的模型权重
  

- 执行切分

  ```
  # 切分模型权重2份，切分好的权重会存放在自定义的output_dir
  bash cut_weight.sh --float 2 --is_gqa
  # 切分模型权重4份
  bash cut_weight.sh --float 4 --is_gqa
  ```

2. **执行模型推理**
- 开启CPU Performance模式以提高模型推理性能（首次开启时，根据提示安装依赖）
  ```
  cpupower frequency-set -g performance
  ```

- 在800I A2执行推理时，可以通过**绑核**以达到最佳性能
  ```
  # 进入./pytorch/examples/atb_speed_sdk/，安装sdk依赖
  cd ../atb_speed_sdk/
  pip install .

  # 进入run.sh，设置环境变量BIND_CPU为1（默认为0，不绑核）
  export BIND_CPU=1
  ```

- 配置必选参数：最大输入输出长度
  修改run.sh中环境变量**MAX_SEQ_LENGTH**为：**期望的最大输入长度 + 最大输出长度**，默认值为2048

- 修改配置参数
当前支持单case推理和多case推理。
multicase=0时，单case推理；
multicase=1时，多case推理；支持用例排列组合，set_case_pair=1时生效。

  ```
  # 双芯模型权重路径
  input_dir="./CodeLlama-34b-Instruct-hf_parallel"
  # 指定芯片，默认为0,1
  device_id=0
  multi_batch_size=[1,4,8,16,32]

  # 单case生效
  seqlen_in=128
  seqlen_out=128
  
  # 多case生效
  # 单case推理(0) or 多case(1)
  multicase=1
  # 多case推理配置参数，默认执行[1,4,8,16,32]的推理
  set_case_pair=0
  # 以下两个变量set_case_pair=0生效，推理默认case，即输入输出分别为[32,64,128,256,512,1024]组合的36组case;
  # 默认输入长度从2^5到2^10
  seqlen_in_range=[5,11]
  # 默认输出长度从2^5到2^10
  seqlen_out_range=[5,11]
  # 以下两个变量set_case_pair=1生效，推理特定case，默认推理(输入长度，输出长度)分别为(256,64),(256,256),(512,512),(1024,1024)4组case;
  seqlen_in_pair=[256,256,512,1024]
  seqlen_out_pair=[64,256,512,1024]
  # LLAMA2-7B or LLAMA2-13B, 为输出文件名字的后缀
  model_name="LLAMA2-7B"
  ```
> 单case: 推理用例为[batch_size, seqlen_in, seqlen_out]；
> 多case: 默认测试batch=1/4/8/16/32，输入32-1024，输出32-1024多case的性能；当set_case_pair=1时，测试seqlen_in_pair/seqlen_out_pair中的用例排列组合；
> 推理完成后性能数据保存在./multibatch_performance_{model_name}_{device_id}.csv，包括用例配置、首token、非首token处理时延等;

- 执行推理
  指令：bash run.sh --[RUN_OPTION] [WORLD_SIZE] [DEVICE_TYPE]  
  ```
  # 800I A2环境执行单卡推理
  bash run.sh --performance 1 d9
  # 800I A2环境执行双卡推理
  bash run.sh --performance 2 d9
  # 300I DUO环境执行单卡双芯推理
  bash run.sh --performance 2 d3
  # 300I DUO环境执行双卡四芯推理
  bash run.sh --performance 4 d3
  ```
  > WORLD_SIZE: 指定芯片数量，实现单卡和多卡推理（默认1）
  > DEVICE_TYPE: d9/d3, 分别适配800I A2和300I DUO芯片型号 (默认d3，支持300I DUO推理)

  该命令会运行一次简单的推理实例warm up，并启动后续的推理；自定义运行可参考`main.py`

## 竞品对比

### 精度(HumanEval)

|          | 910B3               | A100               | 对比              |
|----------|---------------------|--------------------|-------------------| 
| 问题总数  | 164                 | 164                | 1                 |
| 通过总数  | 54                  | 53                 | 1.018867924528302 |
| pass@1   | 0.32926829268292684 | 0.3231707317073171 | 1.018867924528302 |

### 性能

| 设备         | batch_size | 首token推理速度(token/s) | 增量推理速度(token/s) |
|--------------|------------|-------------------------|----------------------|
| 910B3        | 1          | 1.238550074221983       | 8.932253698655556    |
| A100         | 1          | 18.12342392948192       | 16.55063728397924    |
| 对比         | 1          | 0.06833973972253644     | 0.5396924327078233   |

## 精度测试指南

### 1. 下载测试数据集

https://github.com/openai/human-eval/tree/master/data

环境：transformers版本：4.33.1

### 2. 下载测试脚本

下载\human_eval
下载数据脚本：
https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz

1、新建文件夹{CodeLlama-34b-Instruct-hf}，将模型内容上传至该文件夹，切割后的模型放在/CodeLlama-34b-Instruct-hf_test

2、将下列测试文件放在CodeLlama-34b-Instruct-hf_test的上一级目录
--execute.py
--human-eval-v2-20210705.jsonl
--parallel_model_test.py

### 3. 启动测试

>>torchrun --nproc_per_node 2 parallel_model_test.py --model_path={model_path}
可指定端口：--master_port={端口号}
    如：torchrun --nproc_per_node 2 --master_port=29501 parallel_model_test.py --model_path={model_path}

结束后在终端查看测试结果。

### 4. 精度计算

运行后得到评测分数：  
pass@1：human_eval数据集评测分数

