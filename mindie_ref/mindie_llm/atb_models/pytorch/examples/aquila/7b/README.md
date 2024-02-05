# Aquila-7b 模型推理指导[910B]

# 概述

Aquila语言模型是第一个同时支持中英文知识、商业许可协议和遵守国内数据法规的开源语言模型。
Aquila 系列模型的源代码基于 Apache 2.0 协议，而模型权重基于 BAAI Aquila 模型许可协议。只要符合许可限制，用户就可以将其用于商业目的。

- 参考实现：

  ```
    https://huggingface.co/BAAI/Aquila-7B
  ```

# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

| 配套                 | 版本            | 下载链接 |
|--------------------|---------------|------|
| Ascend HDK         | 23.0.RC3.B050 | -    |
| CANN               | 7.0.RC1.B050  | -    |
| python             | 3.8.18        | -    |           
| FrameworkPTAdapter | 5.0.rc3.B050  | -    |

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

安装方法:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run |
| Ascend-hdk-310p-npu-firmware_7.0.t9.0.b221.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装firmwire 以910b为例
chmod +x Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run
./Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run --full
```

##### 1.1.2 安装driver

安装方法：

| cpu     | 包名                                                         | 
|---------|------------------------------------------------------------|
| aarch64 | Ascend-hdk-910b-npu-driver_23.0.rc3.b060_linux-aarch64.run |
| x86     | Ascend-hdk-910b-npu-driver_23.0.rc3.b060_linux-x86_64.run  |
| aarch64 | Ascend-hdk-310p-npu-driver_23.0.rc3.b060_linux-aarch64.run |
| x86     | Ascend-hdk-310p-npu-driver_23.0.rc3.b060_linux-x86-64.run  |

```bash
# 根据CPU架构 以及npu型号 安装对应的 driver
chmod +x Ascend-hdk-910b-npu-driver_23.0.rc3.b060_*.run
./Ascend-hdk-910b-npu-driver_23.0.rc3.b060_*.run --full
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
# 安装 kernel 以910B 为例
chmod +x Ascend-cann-kernels-910b_7.0.T10_linux.run
./Ascend-cann-kernels-910b_7.0.T10_linux.run --install
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
| ttorch-1.11.0-cp310-cp310-linux_aarch64.whl   |
| ttorch-1.11.0-cp38-cp38-linux_aarch64.whl     |
| ttorch-1.11.0-cp39-cp39-linux_aarch64.whl     |
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

1. 下载aquila-7b模型权重，放置到自定义`input_dir`

   ```
   https://huggingface.co/BAAI/Aquila-7B/tree/main
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

若返回True 则使用 cxx11abi=1，否则相反。

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
，拷贝开源模型至工作目录，权重文件（.bin）可采用软链接方式
示例：

```shell
cp ${model_download_path}/*.py ${model_path}/
cp ${model_download_path}/*.json ${model_path}/
cp ${model_download_path}/*.model ${model_path}/
ln -s ${model_download_path}/*.bin ${model_path}/
```

### 安装 atb_speed_sdk

```shell
cd ${llm_path}/pytorch/examples/atb_speed_sdk
pip install .
```

### 接入加速库

- 修改`config.json` , 将 AutoModelForCausalLM 对应的值修改为 "modeling_aquila_fa_rope_model.AquilaForCausalLM"

### 执行推理

修改 `config.ini`
[config文件配置参考](../../atb_speed_sdk/README.md)

```shell
cd ${script_path}
python main.py --task inference
```

main.py 提供了demo推理，精度测试，性能测试三种下游任务。task_name可选inference、precision、performance

**可以使用 MAX_SEQ_LEN 环境变量来设置model支持的最大长度以优化显存占用, 默认使用config里面的max_model_length**
如

```shell
MAX_SEQ_LEN=2048 python main.py --task ${task_name}
```

## 竞品对比

### 精度

| 精度             | 910B3（313T）         | A100                | 对比                    |
|----------------|---------------------|---------------------|-----------------------| 
| STEM           | 0.37209302325581395 | 0.3813953488372093  | -0.009302325581395376 |
| Social Science | 0.5018181818181818  | 0.48363636363636364 | +0.018181818181818188 |
| Humanities     | 0.42023346303501946 | 0.41245136186770426 | +0.007782101167315203 |
| Other          | 0.4088541666666667  | 0.3932291666666667  | +0.015625             |
| Avg acc        | 0.4182763744427935  | 0.4115898959881129  | +0.006686478454680567 |

### 性能

| 芯片型号        | batch_size | 首token推理速度(token/s) | 增量推理速度(token/s)    |
|-------------|------------|---------------------|--------------------|
| A100        | 1          | 22.68088002         | 88.88888889        |
| 910B3（313T） | 1          | 25.425999779824572  | 68.81384251548891  |
| 对比          | 1          | 1.121032330200765   | 0.7741557282895734 |

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

结束后在{ceval_work_dir}/test_result目录下查看测试结果。

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
  model_name=aquila_7b
  perf_mode=detail
  ```

## 3. 执行测试脚本

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
