[TOC]

# InternLM-7B模型-推理指导

# 概述

InternLM开源了一个为实际场景量身定制的70亿参数库模型。该模型具有以下特点：  
-它利用数万亿高质量的令牌进行培训，建立强大的知识库。  
-它为用户提供了一个多功能的工具集，可以灵活地构建自己的工作流。

- 模型权重：

  ```
  https://huggingface.co/internlm/internlm-7b/tree/main
  ```

# 输入输出数据

- 输入数据

| 输入数据           | 大小                                 | 数据类型    | 数据排布格式 | 是否必选 |
|----------------|------------------------------------|---------|--------|------|
| input_ids      | BATCH_SIZE x SEQ_LEN               | INT64   | ND     | 是    |
| attention_mask | BATCH_SIZE x 1 x SEQ_LEN x SEQ_LEN | FLOAT32 | ND     | 否    |

- 输出数据

| 输出数据       | 大小                          | 数据类型  | 数据排布格式 |
|------------|-----------------------------|-------|--------|
| output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64 | ND     |

# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

| 配套             | 版本            | 下载链接 |
|----------------|---------------|------|
| 固件与驱动          | 23.0.RC3.B060 | -    |
| CANN           | 7.0.RC1.B060  | -    |
| python         | 3.8.18        | -    |           
| PytorchAdapter | 1.11.0        | -    |
| 推理引擎           | -             | -    |

**表 2** 推理引擎依赖

| 软件    | 版本要求     |
|-------|----------|
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device |
|---------|--------|
| aarch64 | 800I A2  |
| aarch64 | 310P3  |

# 快速上手

## 获取源码及依赖

### 1. 环境部署

#### 1.1 安装HDK

先安装firmwire，再安装driver

##### 1.1.1 安装firmwire

安装方法:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-800IA2-npu-firmware_7.0.t9.0.b221.run |
| Ascend-hdk-310p-npu-firmware_7.0.t9.0.b221.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装firmwire 以800I A2为例
chmod +x Ascend-hdk-800IA2-npu-firmware_7.0.t9.0.b221.run
./Ascend-hdk-800IA2-npu-firmware_7.0.t9.0.b221.run --full
```

##### 1.1.2 安装driver

安装方法：

| cpu     | 包名                                                     | 
|---------|--------------------------------------------------------|
| aarch64 | Ascend-hdk-800IA2-npu-driver_23.0.rc3.b060_linux-aarch64.run |
| x86     | Ascend-hdk-800IA2-npu-driver_23.0.rc3.b060_linux-x86_64.run  |
| aarch64 | Ascend-hdk-310p-npu-driver_23.0.rc3.b060_linux-aarch64.run |
| x86     | Ascend-hdk-310p-npu-driver_23.0.rc3.b060_linux-x86-64.run |

```bash
# 根据CPU架构 以及npu型号 安装对应的 driver
chmod +x Ascend-hdk-800IA2-npu-driver_23.0.rc3.b060_*.run
./Ascend-hdk-800IA2-npu-driver_23.0.rc3.b060_*.run --full
```

#### 1.2 安装CANN

先安装toolkit 再安装kernel

##### 1.2.1 安装toolkit

安装方法：

| cpu     | 包名                                            |
|---------|-----------------------------------------------|
| aarch64 | Ascend-cann-toolkit_7.0.T10_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_7.0.T10_linux-x86_64.run  |

```bash
# 安装toolkit  以arm为例
chmod +x Ascend-cann-toolkit_7.0.T10_linux-aarch64.run
./Ascend-cann-toolkit_7.0.T10_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

##### 1.2.2 安装kernel

安装方法：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-800IA2_7.0.T10_linux.run |
| Ascend-cann-kernels-310p_7.0.T10_linux.run |

```bash
# 安装 kernel 以800I A2为例
chmod +x Ascend-cann-kernels-800IA2_7.0.T10_linux.run
./Ascend-cann-kernels-800IA2_7.0.T10_linux.run --install
```

#### 1.3 安装PytorchAdapter

先安装torch 再安装torch_npu

##### 1.3.1 安装torch

安装方法：

| 包名                                            |
|-----------------------------------------------|
| torch-1.11.0+cpu-cp38-cp38-linux_x86_64.whl   |
| torch-1.11.0+cpu-cp39-cp39-linux_x86_64.whl   |
| torch-1.11.0+cpu-cp310-cp310-linux_x86_64.whl |
| torch-1.11.0-cp310-cp310-linux_aarch64.whl    |
| torch-1.11.0-cp38-cp38-linux_aarch64.whl      |
| torch-1.11.0-cp39-cp39-linux_aarch64.whl      |
| ...                                           |

根据所使用的环境中的python版本以及cpu类型，选择torch-1.11.0相应的安装包。

```bash
# 安装torch 1.11.0 的python 3.8 的arm版本为例
pip install torch-1.11.0-cp38-cp38-linux_aarch64.whl
```

##### 1.3.2 安装torch_npu

安装方法：

| 包名                           |
|------------------------------|
| pytorch_v1.11.0_py38.tar.gz  |
| pytorch_v1.11.0_py39.tar.gz  |
| pytorch_v1.11.0_py310.tar.gz |
| ...                          |

- 安装选择与torch版本 以及 python版本 一致的npu_torch版本

```bash
# 安装 torch_npu 以torch 1.11.0 的python 3.8的版本为例
tar -zxvf pytorch_v1.11.0_py38.tar.gz
pip install torch*_aarch64.whl
```

### 2. 安装依赖

#### 2.1 推理环境准备

1. 下载internlm-7b模型权重，放置到自定义`input_dir`

   ```
   https://huggingface.co/internlm/internlm-7b/tree/main
   ```

2. 根据版本发布链接，安装加速库

| 加速库包名                                                 |
|-------------------------------------------------------|
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
# 安装atb 
chmod +x Ascend-cann-atb_*.run
./Ascend-cann-atb_*.run --install
source /usr/local/Ascend/atb/set_env.sh
```

3. 根据版本发布链接，解压大模型文件

| 大模型包名                                                                     |
|---------------------------------------------------------------------------|
| Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi1.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi1.tar.gz |

具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

 ```bash
 # 安装大模型加速库
 tar -xzvf Ascend-cann-llm_*.tar.gz
 source set_env.sh
 ```

> 注： 每次运行前都需要 source CANN， 加速库，大模型

## 单芯模型推理

### 拷贝文件

这里假定开源模型下载后的路径为 `{model_path}`
，拷贝开源模型文件夹中的除了 `pytorch_model.bin`和`tokenizer.model`外的所有文件到
{internlm_7b_path}/pytorch/examples/internlm/7b  
示例：

```shell
cd {internlm_7b_path}/pytorch/examples/internlm/7b
cp {model_path}/config.json ./
cp {model_path}/pytorch_model.bin.index.json ./
cp {model_path}/special_tokens_map.json ./
cp {model_path}/tokenization_internlm.py ./
cp {model_path}/tokenizer_config.json ./
cp {model_path}/configuration_internlm.py ./
cp {model_path}/generation_config.json ./
```

- 修改`config.json` , 将 AutoModel 和 AutoModelForCausalLM 对应的值修改为 "
  modeling_internlm_ascend.InternLMForCausalLM"

### 软链接模型权重文件

```shell
ln -s {model_path}/pytorch_model-00001-of-00008.bin pytorch_model-00001-of-00008.bin
ln -s {model_path}/pytorch_model-00002-of-00008.bin pytorch_model-00002-of-00008.bin
ln -s {model_path}/pytorch_model-00003-of-00008.bin pytorch_model-00003-of-00008.bin
ln -s {model_path}/pytorch_model-00004-of-00008.bin pytorch_model-00004-of-00008.bin
ln -s {model_path}/pytorch_model-00005-of-00008.bin pytorch_model-00005-of-00008.bin
ln -s {model_path}/pytorch_model-00006-of-00008.bin pytorch_model-00006-of-00008.bin
ln -s {model_path}/pytorch_model-00007-of-00008.bin pytorch_model-00007-of-00008.bin
ln -s {model_path}/pytorch_model-00008-of-00008.bin pytorch_model-00008-of-00008.bin
ln -s {model_path}/tokenizer.model tokenizer.model
```

### 安装 atb_speed_sdk

```shell
cd ${internlm_7b_path}/pytorch/examples/atb_speed_sdk
pip install .
```

### 配置 config.ini

- 在{internlm_7b_path}/pytorch/examples/internlm/7b目录下创建config.ini文件
  ```shell
  cd {internlm_7b_path}/pytorch/examples/internlm/7b
  vi config.ini
  ```

  [参考atb_speed_sdk下的README.md](../../atb_speed_sdk/README.md)

- 复制readme中”配置文件样例“章节的内容，以config.ini文件名保存

- 配置config.ini中属性值：
  ```
  model_path={internlm_7b_path}/pytorch/examples/internlm/7b
  work_dir={internlm_7b_path}/pytorch/examples/internlm/7b/script
  ```

### 执行推理

```
python main.py --task inference
```

该命令会运行一次简单的推理实例warm up，并启动后续的1个问答

# 竞品对比

## 未接入FA、Rope

### 精度

| 精度             | 800I A2（313T）      | A100               | 对比                 |
|----------------|--------------------|--------------------|--------------------| 
| STEM           | 0.4558139534883721 | 0.4511627906976744 | 1.010309278350516  |
| Social Science | 0.5963636363636363 | 0.6036363636363636 | 0.9879518072289156 |
| Humanities     | 0.5797665369649806 | 0.5836575875486382 | 0.9933333333333333 |
| Other          | 0.4739583333333333 | 0.4739583333333333 | 1                  |
| Avg acc        | 0.513372956909361  | 0.5141158989598811 | 0.9985549132947976 |

### 性能

| 芯片型号        | batch_size | 首token推理速度(token/s) | 增量推理速度(token/s)    |
|-------------|------------|---------------------|--------------------|
| A100        | 1          | 8.472709848091108   | 32.70245536545943  |
| 800I A2（313T） | 1          | 10.62104062813386   | 32.45752413727395  |
| 对比          | 1          | 1.253558875325675   | 0.9925103107565379 |

## 接入FA、Rope

### 精度

| 精度             | 800I A2（313T）         | A100               | 对比                 |
|----------------|---------------------|--------------------|--------------------|
| STEM           | 0.45813953488372094 | 0.4511627906976744 | 1.0154639175257734 |
| Social Science | 0.5963636363636363  | 0.6036363636363636 | 0.9879518072289156 |
| Humanities     | 0.5797665369649806  | 0.5836575875486382 | 0.9933333333333334 |
| Other          | 0.4713541666666667  | 0.4739583333333333 | 0.9945054945054946 |
| Avg acc        | 0.513372956909361   | 0.5141158989598811 | 0.9985549132947977 |

### 性能

| 芯片型号        | batch_size | 首token推理速度(token/s) | 增量推理速度(token/s)    |
|-------------|------------|---------------------|--------------------|
| FT A100     | 1          | 22.68088002         | 88.88888889        |
| 800I A2（313T） | 1          | 23.01697799230084   | 66.73531056262117  |
| 对比          | 1          | 1.0148185595975319  | 0.7507722438201034 |

# 附录：

# 精度测试指南

## 配置说明

参考 [SDK精度测试指南CEVAL章节](../../atb_speed_sdk/README.md)

## 运行脚本

- 单芯

```shell
cd ${script_path}
python main.py --task precision
```

- 多芯

```shell
cd ${script_path}
bash cut_model_and_run.sh precision
```

结束后在${ceval_work_dir}/test_result目录下查看测试结果。[双芯结果每个两份，只需看其中一份即可]。

| 文件                        | 用途                   | 
|---------------------------|----------------------| 
| device0.log               | 运行过程日志               |
| cache0.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| result_0_classes_acc.json | 测试数据下按不同维度统计准确率      |
| result_0_subject_acc.json | 测试数据下按不同学科统计准确率      |

**注意：后续重新运行， 需要删除当前目录下生成的test_result文件夹，否则只会读取当前的目录下的测试结果**

# 性能测试

在功能运行正常的基础下，执行以下步骤进行性能测试

## 按照推理指导,下载模型及配置路径，并安装atb_speed_sdk

## 1. 准备

参考 [SDK性能测试指南精确打点法章节](../../atb_speed_sdk/README.md) 进行准备

## 2. 修改配置文件

- 配置config.ini中[performance]属性， 如下：
  ```
  model_name=internlm_7b
  perf_mode=detail
  ```

## 3. 执行测试脚本

- 单芯

```shell
cd ${script_path}
RETURN_PERF_DETAIL=1 python main.py --task performance
```

为了不影响正常使用，将`RETURN_PERF_DETAIL`设置成1来返回具体的性能测试的值，默认是0

### 性能测试结果

得到性能测试结果csv `performance_test_npu_${model_name}_xxx.csv`

### 结果分析

| 列名                            | 含义         |
|-------------------------------|------------|
| batch_size                    | batch大小    |
| input_seq_len(Encoding)       | 输入长度       |
| output_seq_len(Decoding)	     | 输出长度       |
| ResponseTime(s)	              | 总响应时间      |
| forward_first_token_time(ms)  | 首token推理时长 |
| forward_next_token_time(ms)   | 增量推理时长     |
| pre_next_token_time(ms)	      | 前处理时长      |
| post_next_token_time_post(ms) | 后处理时长      |

