[TOC]

# BaiChuan-7B模型-推理指导

# 概述

Baichuan-7B是由百川智能开发的一个开源的大规模预训练模型。基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。在标准的中文和英文权威benchmark（C-EVAL/MMLU）上均取得同尺寸最好的效果。

- 参考实现：

  ```
  https://huggingface.co/baichuan-inc/Baichuan-7B
  ```

# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

| 配套                 | 版本            | 下载链接 |
|--------------------|---------------|------|
| Ascend HDK         | 23.0.RC3.B100 |      |
| CANN               | 7.0.RC1.B100  |      |
| python             | 3.9.18        |      |           
| FrameworkPTAdapter | 5.0.RC3.B090  |      |

**表 2** 推理引擎依赖

| 软件    | 版本要求     |
|-------|----------|
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device |
|---------|--------|
| aarch64 | 910B3  |
| aarch64 | 310P3  |

# 快速上手

## 获取源码及依赖

### 1. 环境部署

#### 1.1 安装HDK

先安装firmwire，再安装driver

##### 1.1.1 安装firmwire

安装方法: xxx代表具体版本

| 包名                                   |
|--------------------------------------|
| Ascend-hdk-910b-npu-firmware_xxx.run |
| Ascend-hdk-310p-npu-firmware_xxx.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装firmwire
chmod +x Ascend-hdk-310p-npu-firmware_xxx.run
./Ascend-hdk-310p-npu-firmware_xxx.run --full
```

##### 1.1.2 安装driver

安装方法：

| cpu     | 包名                                               | 
|---------|--------------------------------------------------|
| aarch64 | Ascend-hdk-910b-npu-driver_xxx_linux-aarch64.run |
| x86     | Ascend-hdk-910b-npu-driver_xxx_linux-x86_64.run  |
| aarch64 | Ascend-hdk-310p-npu-driver_xxx_linux-aarch64.run |
| x86     | Ascend-hdk-310p-npu-driver_xxx_linux-x86-64.run  |

```bash
# 根据CPU架构 以及npu型号 安装对应的 driver
chmod +x Ascend-hdk-310p-npu-driver_23.0.rc3.b060_*.run
./Ascend-hdk-310p-npu-driver_23.0.rc3.b060_*.run --full
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
| Ascend-cann-kernels-910b_7.0.T10_linux.run |
| Ascend-cann-kernels-310p_7.0.T10_linux.run |

```bash
# 安装 kernel 以310P 为例
chmod +x Ascend-cann-kernels-310p_7.0.T10_linux.run
./Ascend-cann-kernels-310p_7.0.T10_linux.run --install
```

#### 1.3 安装PytorchAdapter

先安装torch 再安装torch_npu

##### 1.3.1 安装torch

安装方法：

| 包名                                           |
|----------------------------------------------|
| torch-2.0.1+cpu-cp38-cp38-linux_x86_64.whl   |
| torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl   |
| torch-2.0.1+cpu-cp310-cp310-linux_x86_64.whl |
| torch-2.0.1-cp310-cp310-linux_aarch64.whl    |
| torch-2.0.1-cp38-cp38-linux_aarch64.whl      |
| torch-2.0.1-cp39-cp39-linux_aarch64.whl      |
| ...                                          |

根据所使用的环境中的python版本以及cpu类型，选择torch-2.0.1相应的安装包。

```bash
# 安装torch 2.0.1 的python 3.9 的arm版本为例
pip install torch-2.0.1-cp39-cp39-linux_aarch64.whl
```

##### 1.3.2 安装torch_npu

安装方法：

| 包名                          |
|-----------------------------|
| pytorch_v2.0.1_py38.tar.gz  |
| pytorch_v2.0.1_py39.tar.gz  |
| pytorch_v2.0.1_py310.tar.gz |
| ...                         |

- 安装选择与torch版本 以及 python版本 一致的npu_torch版本

```bash
# 安装 torch_npu 以torch 2.0.1 的python 3.9的版本为例
tar -zxvf pytorch_v2.0.1_py39.tar.gz
pip install torch*_aarch64.whl
```

#### 1.3.3 requirements

| 包名            | 推荐版本   |  
|---------------|--------|
| transformers  | 4.30.2 | 
| decorator     | 5.1.1  |
| sympy         | 1.11.1 |
| scipy         | 1.11.3 |
| attrs         | 23.1.0 |
| psutil        | 5.9.6  |
| sentencepiece | 0.1.99 |

### 2. 安装依赖

#### 路径变量解释

| 变量名                 | 含义                                                                   |  
|---------------------|----------------------------------------------------------------------|
| model_download_path | 开源权重放置目录                                                             | 
| llm_path            | 加速库及模型库下载后放置目录                                                       |
| model_path          | 工作时模型所在的目录，可以和model_download_path相同，但一般模型是公共的，为了避免影响其他用户，单独建一个模型工作目录 |
| script_path         | 工作脚本所在路径，本文为${llm_path}/pytorch/examples/baichuan/7b                |
| ceval_work_dir      | ceval数据集、及结果保存所在目录，不必和模型脚本在相同目录                                      |

#### 2.1 推理环境准备

1. 下载baichuan_7b模型权重，放置到自定义`${model_download_path}` (请下载链接中'Files and versions'页签下的所有文件)

   ```
   https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main
   ```

2. 根据版本发布链接，安装加速库
   将加速库下载至 `${llm_path}` 目录

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

3. 根据版本发布链接，安装加速库
   将加速库下载至 `${llm_path}` 目录

| 大模型包名                                                             |
|-------------------------------------------------------------------|
| Ascend-cann-llm_{version_id}_linux-x86_64_torch2.0.1-abi0.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-x86_64_torch2.0.1-abi1.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch2.0.1-abi0.tar.gz |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch2.0.1-abi1.tar.gz |

具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

 ```bash
 # 安装大模型加速库
 cd ${llm_path}
 tar -xzvf Ascend-cann-llm_*.tar.gz
 source set_env.sh
 ```

4. 下载CEval数据集

   若需执行精度测试，请参考附录中的精度测试指南 进行下载

5. 设置环境变量

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/atb/set_env.sh
   source ${llm_path}/set_env.sh
   ```
   > 注： 每次运行前都需要 source CANN， 加速库，大模型

### 拷贝文件

### 准备

#### 1. 将开源模型拷贝到模型工作目录，bin文件使用软链接即可,同时将modeling文件拷贝到模型，并修改开源的config.json,

```shell
cp ${model_download_path}/*.py ${model_path}/
cp ${model_download_path}/*.json ${model_path}/
cp ${model_download_path}/*.model ${model_path}/
cp -s ${model_download_path}/*.bin ${model_path}/
```

#### 2. 安装 atb_speed_sdk

```shell
cd ${llm_path}/pytorch/examples/atb_speed_sdk
pip install .
```

#### 3. 张量并行模型切分（仅在模型需要多卡并行时使用）

```shell
cp ${script_path}/modeling_baichuan_cut.py ${model_path}
```

修改 ${model_path}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_baichuan_cut.BaiChuanForCausalLM"`

```text
修改`${script_path}/cut_model_and_run_baichuan.sh`    
将 `input_dir` 修改为模型所在路径 `${model_path}` 
将 `output_dir` 修改为切分后的模型所存储的路径,比如仍为原目录 `${model_path}`。模型切分成功后，会自动生成新目录part_model(用户无需新建该文件夹)，即：${model_path/part_model}
将 `world_size_` 修改成希望切成的卡的数量
```

执行

```shell
cd ${script_path}
bash cut_model_and_run.sh
```

切分所需时间较长，切分完成后，将会打印 'Tensor parallelism weights have been successfully saved.'。

#### 4. 拷贝运行用的modeling （仅在模型需要多卡并行时使用）

#### 上面切分时会自动做以下操作

*如果需要执行别的modeling则需要修改以下文件*  
rank_id，表示卡的编号，0,1,2,3.。。

```shell
cd ${model_path}/part_model/{rank_id}
cp ${script_path}/modeling_baichuan_ascend.py ./
```

修改
${model_path}/part_model/{rank_id}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_baichuan_ascend.BaiChuanForCausalLM"`

### 执行推理

#### 修改 ${script_path}/config.ini

[config文件配置参考](../../atb_speed_sdk/README.md)

#### main.py

提供了demo推理，精度测试，性能测试三种下游任务。task_name可选inference、precision、performance

- 单芯
  修改 ${model_path}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_baichuan_ascend.BaiChuanForCausalLM"`

```shell
python main.py --task ${task_name}
```

- 多芯

```shell
bash cut_model_and_run.sh ${task_name}
```

**可以使用 MAX_SEQ_LEN 环境变量来设置model支持的最大长度以优化显存占用, 默认使用config里面的max_model_length**  
如

```shell
MAX_SEQ_LEN=2048 python main.py --task ${task_name}
```

或

```shell
MAX_SEQ_LEN=2048 bash cut_model_and_run.sh ${task_name}
```


# 附录

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
  model_name=baichuan_7b
  perf_mode=detail
  ```

## 3. 执行测试脚本

- 单芯

```shell
cd ${script_path}
TIMEIT=1 python main.py --task performance
```

- 多芯

```shell
cd ${script_path}
TIMEIT=1 bash cut_model_and_run.sh performance
```

为了不影响正常使用，将`TIMEIT`设置成1来返回具体的性能测试的值，默认是0

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

