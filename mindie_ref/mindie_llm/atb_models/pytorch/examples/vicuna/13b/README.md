#  Vicuna-13B模型-推理指导（800I A2）

- [Vicuna-13B模型-推理指导（800I A2）](#vicuna-13b模型-推理指导800i-a2)
- [概述](#概述)
- [版本配套](#版本配套)
- [快速上手](#快速上手)
  - [获取源码及依赖](#获取源码及依赖)
  - [推理环境准备](#推理环境准备)
  - [模型推理](#模型推理)
- [模型推理精度](#模型推理精度)

# 概述

Vicuna是由 LMSYS 发布的基于Llama 2用ShareGPT收集的125K对话集微调的大模型，最长可以支持16K

- 参考实现：

  ```
  https://github.com/lm-sys/FastChat
  ```


# 版本配套

该模型需要以下插件与驱动

**表 1** 版本配套表

| 配套           | 版本          | 下载链接 |
| -------------- | ------------- | -------- |
| 固件与驱动     | 24.0.T1.B010 | -        |
| CANN           | 8.0.T2.B010  | -        |
| Python         | 3.9.18        | -        |
| torch         | 2.0.1        | -        |
| PytorchAdapter | 6.0.RC1.B010        | -        |


# 快速上手

## 获取源码及依赖

1. 环境部署

- 1.1. 安装HDK

> 先安装firmwire，再安装driver

  1.1.1. 安装firmwire

  安装方法:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-910b-npu-firmware_7.2.t3.0.b023.run |

  ```bash
  # 安装firmwire
  chmod +x Ascend-hdk-910b-npu-firmware_7.2.t3.0.b023.run
  ./Ascend-hdk-910b-npu-firmware_7.2.t3.0.b023.run --full
  ```

  1.1.2. 安装driver

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-hdk-910b-npu-driver_24.0.rc1.b010_linux-aarch64.run |
| x86     | Ascend-hdk-910b-npu-driver_24.0.rc1.b010_linux-x86-64.run |

  ```bash
  # 根据CPU架构安装对应的 driver
  chmod +x Ascend-hdk-910b-npu-driver_24.0.rc1.b010_*.run
  ./Ascend-hdk-910b-npu-driver_24.0.rc1.b010_*.run --full
  ```

- 1.2. 安装CANN

> 先安装toolkit 再安装kernel

  1.2.1. 安装toolkit

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-cann-toolkit_8.0.T2_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_8.0.T2_linux-x86_64.run |

  ```bash
  # 安装toolkit
  chmod +x Ascend-cann-toolkit_8.0.T2_linux-*.run
  ./Ascend-cann-toolkit_8.0.T2_linux-*.run --install
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
  1.2.2. 安装kernel

  安装方法：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-910b_8.0.T2_linux.run |

  ```bash
  # 安装 kernel
  chmod +x Ascend-cann-kernels-910b_8.0.T2_linux.run
  ./Ascend-cann-kernels-910b_8.0.T2_linux.run --install
  ```

- 1.3. 安装PytorchAdapter

> 安装apex、torch、torch_npu

  1.3.1 安装torch

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | torch-2.0.1-cp39-cp39-linux_aarch64.whl |
| x86     | torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl |

  根据所使用的环境中的python版本，选择torch-2.0.1相应的安装包。

  ```bash
  # 安装torch 2.0.1 的python 3.9 的arm版本为例
  pip install torch-2.0.1-cp39-cp39-linux_aarch64.whl
  ```

  1.3.2 安装torch_npu

  安装方法：

| 包名                          |
|-----------------------------|
| pytorch_v2.0.1_py39.tar.gz |

> 安装选择与torch版本 以及 python版本 一致的torch_npu版本

  ```bash
  # 安装 torch_npu 以torch 2.0.1 的python 3.9的arm版本为例
  tar -zxvf pytorch_v2.0.1_py39.tar.gz
  pip install torch*_aarch64.whl
  ```

## 推理环境准备

> 安装配套软件。安装python依赖。

  ```
  pip3 install -r requirements.txt
  ```

1. 下载vicuna-13b-v1.5-16k模型权重，放置到自定义`input_dir`

   ```
   https://huggingface.co/lmsys/vicuna-13b-v1.5-16k
   ```

2. 根据版本发布链接，安装加速库 

   | 加速库包名                                            |
   | ----------------------------------------------------- |
   | Ascend-cann-atb_{version}_cxx11abi0_linux-aarch64.run |
   | Ascend-cann-atb_{version}_cxx11abi1_linux-aarch64.run |
   | Ascend-cann-atb_{version}_cxx11abi1_linux-x86_64.run  |
   | Ascend-cann-atb_{version}_cxx11abi0_linux-x86_64.run  |
   
   具体使用cxx11abi0 还是cxx11abi1 可通过python命令查询
   
   ```python
   import torch

   torch.compiled_with_cxx11_abi()
   ```
   
   若返回True 则使用 cxx11abi1，否则相反。
   
   ```bash
   # 安装
   chmod +x Ascend-cann-atb_7.0.T10_*.run
   ./Ascend-cann-atb_7.0.T10_*.run --install
   source /usr/local/Ascend/atb/set_env.sh
   ```
   
3. 根据版本发布链接，解压大模型文件

   | 大模型包名                                                   |
   | ------------------------------------------------------------ |
   | Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz |
   | Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi1.tar.gz |
   | Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz |
   | Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi1.tar.gz |

    具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

   ```bash
   # 安装
   mkdir {llm_path}
   tar -xzvf Ascend-cann-llm_*.tar.gz -C {llm_path} --no-same-owner
   source set_env.sh
   ```

   > 注： 每次运行前都需要 source CANN， 加速库，大模型

## 模型推理
1. 如果跑多卡多芯推理，需要先切分模型权重，切分方法如下：

- 修改代码

  1. 修改`cut_weight.sh`中`input_dir`为实际存放模型权重的路径
  
  2. 修改`cut_weight.sh`中`output_dir`为自定义路径，用于存放切分后的模型权重

- 执行切分

  ```
  # 切分模型权重2份，切分好的权重会存放在自定义的output_dir
  bash cut_weight.sh --float 2
  # 切分模型权重8份
  bash cut_weight.sh --float 8
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

  # 进入sdk_config.ini，设置环境变量bind_cpu为1（默认为1，绑核）
  bind_cpu=1
  ```

- 配置必选参数：最大输入输出长度
  修改run_sdk_test.sh中环境变量**MAX_SEQ_LENGTH**为：**期望的最大输入长度 + 最大输出长度**，默认值为2048

- 修改sdk_config.ini配置参数

  ```
  # [model]
  # 模型权重路径
  model_path="./vicuna-13b-v1.5-16k_parallel_8
  # 指定芯片，默认为0,1
  device_id=0,1,2,3,4,5,6,7

  # [performance]
  # 性能测试模型名称，用于结果文件的命名
  model_name=vicuna_13b_v1.5_16k
  # 测试的batch size
  batch_size=1
  # 测试的输入的最大2的幂
  max_len_exp=10
  # 测试的输入的最小2的幂
  min_len_exp=5
  # 特定用例测试，格式为[[seq_in,seq_out]]，注意当设置这个参数时，max_len_exp min_len_exp不生效
  case_pair=[[256,256],[512,512]]
  
> 推理完成后性能数据保存在./performance_test_{model_name}_{device_type}_bs{batch_size}.csv，包括用例配置、端到端时间、首token、非首token处理时延等;

- 执行推理
  指令：bash run_sdk_test.sh [WORLD_SIZE] [DEVICE_TYPE] [RUN_OPTION]
  ```
  bash run_sdk_test.sh 8 d9 performance
  ```
  > WORLD_SIZE: 指定芯片数量，实现单卡和多卡推理（默认1）
  > DEVICE_TYPE: d9/d3, 分别适配800I A2和300I DUO芯片型号 (默认d3，支持300I DUO推理)
  > RUN_OPTION: run，部分case推理；performance，性能测试；precision，精度测试


# MMLU数据集精度验证指南
> 采用5-shot的方式验证模型推理精度。 

## 1.下载MMLU数据集
  ```
  wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
  tar -xvf data.tar
  ```

## 2. 安装atb-speed插件
  ```
  cd ../atb_speed_sdk/
  pip install .
  ```

## 3.配置精度测试参数
1. 在当前目录新建工作文件夹${mmlu_test_dir}
2. 将下载的测试数据集进行解压后的数据放置在${mmlu_test_dir}
3. 修改sdk_config.ini文件中精度测试的相关配置，设置模型路径、工作目录、device id(默认0、1卡)、和batch size(默认1)
    * model_path=./vicuna-13b-v1.5-16k_parallel_8
    * work_dir=${mmlu_test_dir}
    * device=0,1,2,3,4,5,6,7
    * batch=1

目录结构示例:  
    --test_result 跑完之后生成  
    --data (包含：数据文件夹dev、test、val三者)

## 4. 运行并查看结果

**4.1 开始精度数据集推理**
  ```
  bash run_sdk_test.sh 8 d9 precision
  ```

**4.2 查看结果**
| test_result目录                        | 用途                   | 
|---------------------------|----------------------| 
| cache.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| result_classes_acc.json  | 测试数据下按不同维度统计准确率      |
| result_subject_acc.json  | 测试数据下按不同学科统计准确率      |

> 注：开始下一次数据集精度推理前，请重命名之前保存的结果文件夹 ${mmlu_test_dir}/test_result


# CEVAL数据集精度验证指南
> 采用5-shot的方式验证模型推理精度。 

## 1.下载CEVAL数据集
  ```
  wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
  unzip ceval-exam.zip -d data
  ```

## 2. 安装atb-speed插件
  ```
  cd ../atb_speed_sdk/
  pip install .
  ```

## 3.配置精度测试参数
1. 在当前目录新建工作文件夹${ceval_test_dir}
2. 将下载的测试数据集进行解压后的数据放置在${ceval_test_dir}
3. 修改sdk_config.ini文件中精度测试的相关配置，设置模型路径、工作目录、device id(默认0、1卡)、和batch size(默认1)
    * model_path=./vicuna-13b-v1.5-16k_parallel_8
    * work_dir=${ceval_test_dir}
    * device=0,1,2,3,4,5,6,7
    * batch=1

目录结构示例:  
    --test_result 跑完之后生成  
    --data (包含：数据文件夹dev、test、val三者)

## 4. 运行并查看结果

**4.1 开始精度数据集推理**
  ```
  bash run_sdk_test.sh 8 d9 precision
  ```

**4.2 查看结果**
| test_result目录                        | 用途                   | 
|---------------------------|----------------------| 
| cache.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| result_classes_acc.json  | 测试数据下按不同维度统计准确率      |
| result_subject_acc.json  | 测试数据下按不同学科统计准确率      |

> 注：开始下一次数据集精度推理前，请重命名之前保存的结果文件夹 ${ceval_test_dir}/test_result


# 模型推理精度

| Vicuna 5-shot | MMLU | CEVAL |
| ------------------ | ------------- | ------------- |
| Average(%) | 55.69 | 40.19	|