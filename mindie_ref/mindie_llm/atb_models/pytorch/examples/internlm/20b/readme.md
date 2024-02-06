[TOC]

# InternLM-20B模型-推理指导

# 概述

上海人工智能实验室与SenseTime科技、香港中文大学、复旦大学合作，正式发布了200亿参数预训练模型InternLM-20B。InternLM-20B在超过2.3T的包含高质量英文、中文和代码数据的Token上进行了预训练。此外，Chat版本还经过了SFT和RLHF培训，使其能够更好、更安全地满足用户的需求。

在模型结构方面，InternLM-20B选择了更深的架构，深度设置为60层。这超过了使用32或40层的传统20b和13B模型。当参数有限时，增加层数可以增强模型的整体能力。此外，与InternLM-20b相比，InternLM-20B使用的预训练数据经过了更高质量的清洗，并补充了知识丰富的数据，旨在加强理解和推理能力。因此，它在理解、推理、数学和编程能力方面表现出了显著的改进——所有这些都考验着语言模型的技术熟练程度。总体而言，InternLM-20B具有以下特点：

整体表现突出
强大的工具调用能力
支持16k上下文长度（通过推断外推）
更好的价值对齐。

- 模型权重：

  ```
  https://huggingface.co/internlm/internlm-20b/tree/main
  ```

- 模型代码commit id：

  ```
  c56a72957239b490ea206ea857e86611b3f65f3a
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

| 配套             | 版本                            | 下载链接 |
|----------------|-------------------------------|------|
| 固件与驱动          | 23.0.rc3                      | -    |
| CANN           | FrameworkPTAdapter 5.0.0.B050 | -    |
| python         | 3.9.18                        | -    |           
| PytorchAdapter | 2.0.1                         | -    |
| 推理引擎           | -                             | -    |

**表 2** 推理引擎依赖

| 软件    | 版本要求     |
|-------|----------|
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device |
|---------|--------|
| aarch64 | 910B4  |

# 快速上手

## 获取源码及依赖

### 1. 环境部署

#### 1.1 安装HDK

先安装firmwire，再安装driver

##### 1.1.1 安装firmwire

##### 1.1.2 安装driver

#### 1.2 安装CANN

先安装toolkit 再安装kernel

##### 1.2.1 安装toolkit

##### 1.2.2 安装kernel

#### 1.3 安装PytorchAdapter

先安装torch 再安装torch_npu

##### 1.3.1 安装torch

##### 1.3.2 安装torch_npu

##### 1.3.3 安装依赖

| 包名            | 推荐版本   |  
|---------------|--------|
| transformers  | 4.30.2 | 
| decorator     | 5.1.1  |
| sympy         | 1.11.1 |
| scipy         | 1.11.3 |
| attrs         | 23.1.0 |
| psutil        | 5.9.6  |
| sentencepiece | 0.1.99 |
| pandas        | 2.1.4  |

### 2. 安装依赖

#### 2.1 推理环境准备

1. 下载internlm-20b模型权重，放置到自定义`input_dir`

   ```
   https://huggingface.co/internlm/internlm-20b/tree/main
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

| 大模型包名                                    |
|------------------------------------------|
| Ascend-cann-transformer-llm_abi_1.tar.gz |
| Ascend-cann-transformer-llm_abi_0.tar.gz |

具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

 ```bash
 # 安装大模型加速库
 tar -xzvf Ascend-cann-transformer-llm_abi*.tar.gz
 source set_env.sh
 ```

> 注： 每次运行前都需要 source CANN， 加速库，大模型

## 单芯模型推理

### 拷贝文件

这里假定开源模型下载后的路径为 `{model_path}`
，拷贝开源模型文件夹中的除了 `pytorch_model.bin`和`tokenizer.model`外的所有文件到
{internlm_20b_path}/pytorch/examples/internlm/20b  
示例：

```shell
cd {internlm_20b_path}/pytorch/examples/internlm/20b
mkdir model
cd model
cp {model_path}/config.json ./
cp {model_path}/pytorch_model.bin.index.json ./
cp {model_path}/special_tokens_map.json ./
cp {model_path}/tokenization_internlm.py ./
cp {model_path}/tokenizer_config.json ./
cp {model_path}/configuration_internlm.py ./
cp {model_path}/generation_config.json ./
cp {model_path}/.gitattributes ./
```

- 修改`config.json` , 将 AutoModel 和 AutoModelForCausalLM 对应的值修改为 "
  modeling_internlm_fa_rope_parallel_model.InternLMForCausalLM"

### 软链接模型权重文件

```shell
ln -s {model_path}/pytorch_model-00001-of-00005.bin pytorch_model-00001-of-00005.bin
ln -s {model_path}/pytorch_model-00002-of-00005.bin pytorch_model-00002-of-00005.bin
ln -s {model_path}/pytorch_model-00003-of-00005.bin pytorch_model-00003-of-00005.bin
ln -s {model_path}/pytorch_model-00004-of-00005.bin pytorch_model-00004-of-00005.bin
ln -s {model_path}/pytorch_model-00005-of-00005.bin pytorch_model-00005-of-00005.bin
ln -s {model_path}/tokenizer.model tokenizer.model
```

### 安装 atb_speed_sdk

```shell
cd ${internlm_20b_path}/pytorch/examples/atb_speed_sdk
pip install .
```

#### 3. 张量并行模型切分（仅在模型需要多卡并行时使用）

```shell
cp ${internlm_20b_path}/pytorch/examples/internlm/20b/modeling_internlm_cut.py ${model_path}
```

修改 ${model_path}里的config.json中的kv对，改成
`"AutoModel": "modeling_internlm_cut.InternLMForCausalLM"`
`"AutoModelForCausalLM": "modeling_internlm_cut.InternLMForCausalLM"`

```text
执行：python ${internlm_20b_path}/pytorch/examples/internlm/20b/cut_model_util.py --input_path={} --output_path={}
其中
`input_path` 为模型所在路径 `${model_path}` 
`output_path` 为切分后的模型所存储的路径,比如 `${model_path}/part_model`
```

切分所需时间较长，切分完成后，将会打印 'Tensor parallelism weights have been successfully saved.'。

#### 拷贝运行用的modeling （仅在模型需要多卡并行时使用）

rank_id，表示卡的编号，0,1,2,3.。。，并修改里面的config.json

```shell
cd ${model_path}/part_model/{rank_id}
cp ${internlm_20b_path}/pytorch/examples/internlm/20b/modeling_internlm_fa_rope_parallel_model.py ./
vim config.json
```

修改config.json中的kv对，改成
`"AutoModelForCausalLM": "modeling_internlm_fa_rope_parallel_model.InternLMForCausalLM"`
`"bias": true`

### 配置 config.ini

- 在{internlm_20b_path}/pytorch/examples/internlm/20b目录下创建config.ini文件
  ```shell
  cd {internlm_20b_path}/pytorch/examples/internlm/20b/script
  vi config.ini
  ```

  [参考atb_speed_sdk下的README.md](../../atb_speed_sdk/README.md)

- 复制readme中”配置文件样例“章节的内容，以config.ini文件名保存

- 配置config.ini中属性值：
  ```
  model_path={output_path}
  work_dir={internlm_20b_path}/pytorch/examples/internlm/20b/script
  ```

# 竞品对比

## 接入FA、Rope

### 精度

| 精度             | 910B4              | A30                | 对比                |
|----------------|--------------------|--------------------|-------------------| 
| STEM           | 0.5232558139534884 | 0.5232558139534884 | 1                 |
| Social Science | 0.6727272727272727 | 0.6727272727272727 | 1                 |
| Humanities     | 0.6381322957198443 | 0.6381322957198443 | 1                 |
| Other          | 0.5572916666666666 | 0.5546875          | 1.004694835680751 |
| Avg acc        | 0.5854383358098069 | 0.5846953937592868 | 1.001270648030496 |

### 性能

| 芯片型号             | batch_size | 首token推理速度(token/s) | 增量推理速度(token/s) |
|------------------|------------|---------------------|-----------------|
| A30(llama 13B换算) | 1          | 8.05552113          | 31.88619083     |
| 910B4            | 1          | 8.813317838         | 25.80395639     |
| 对比               | 1          | 1.094071717         | 0.809251771     |

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

```
torchrun --nproc_per_node 2 main.py --task precision
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
  model_name=internlm_20b
  perf_mode=detail
  ```

## 3. 执行测试脚本

```
RETURN_PERF_DETAIL=1 torchrun --nproc_per_node 2 main.py --task performance
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

