#  LLaMA-7B模型-推理指导

# 概述

LLaMA（Large Language Model Meta AI），由 Meta AI 发布的一个开放且高效的大型基础语，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 参考实现：

  ```
  https://github.com/facebookresearch/llama
  ```

# 输入输出数据

- 输入数据

  | 输入数据       | 大小                               | 数据类型 | 数据排布格式 | 是否必选 |
  | -------------- | ---------------------------------- | -------- | ------------ | -------- |
  | input_ids      | BATCH_SIZE x SEQ_LEN               | INT64    | ND           | 是       |
  | attention_mask | BATCH_SIZE x 1 x SEQ_LEN x SEQ_LEN | FLOAT32  | ND           | 否       |

- 输出数据

  | 输出数据   | 大小                        | 数据类型 | 数据排布格式 |
  | ---------- | --------------------------- | -------- | ------------ |
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64    | ND           |

# 推理环境准备

该模型需要以下插件与驱动

**表 1** 版本配套表

| 配套           | 版本          | 下载链接 |
| -------------- | ------------- | -------- |
| 固件与驱动     | 23.0.RC3.B060 | -        |
| CANN           | 7.0.RC1.B060  | -        |
| Python         | 3.9.11        | -        |
| PytorchAdapter | 1.11.0        | -        |
| 推理引擎       | -             | -        |

**表 2** 推理引擎依赖

| 软件  | 版本要求 |
| ----- | -------- |
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device |
| ------- | ------ |
| aarch64 | 300I   |
| x86     | 300I   |

# 快速上手

## 获取源码及依赖

1. 环境部署

- 安装HDK

- 安装CANN

- 安装PytorchAdapter

- 安装依赖

  参考

  推理环境准备

  安装配套软件。安装python依赖。

  ```
  pip3 install -r requirements.txt
  ```

1. 下载LLaMA-7B模型权重，放置到自定义`input_dir`

   ```
    https://huggingface.co/NousResearch/Llama-2-7b-hf
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

   | 大模型包名                                                  |
   | ----------------------------------------------------------- |
   | Ascend-cann-transformer-llm_abi_0-pta_1.11.0-aarch64.tar.gz |
   | Ascend-cann-transformer-llm_abi_1-pta_1.11.0-aarch64.tar.gz |

    具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

   ```bash
   # 安装
   # cd {llm_path}
   tar -xzvf Ascend-cann-transformer-llm_abi*.tar.gz
   source set_env.sh
   ```

   > 注： 每次运行前都需要 source CANN， 加速库，大模型

## 模型推理

- 修改代码

  1. 根据芯片类型是910还是310拷贝对应的modeling文件到 到对应的transformer库路径下，
  如果是910芯片，拷贝modeling_llama_910.py文件
  如果是310芯片，拷贝modeling_llama_310.py
  示例：

    ```bash
  cp modeling_llama_910.py /usr/local/python3.9/site-packages/transformers/models/llama/modeling_llama.py
    ```

- 2.修改run_llama_example.py文件里的模型权重路径
  


- 3.执行推理

  ```bash
  python run_llama_example.py
  ```

  该命令会运行一次简单的推理实例warm up，并启动后续的1个问答

- 自定义运行可参考`run_llama_example.py`

或者直接执行bash run.sh [--run|--performance|--precision] [model script path] [device id] [310 | 910]"

配置可选参数：最大输入输出长度(Optional)
默认值为2048，可以根据用户需要, 在脚本中手动配置最大输入输出长度，把modeling_llama_modelv2.py脚本中的变量MAX_SEQ_LENGTH改为：期望的最大输入长度 + 最大输出长度

decoder-only结构默认tokenlizer时应用左padding，当前910暂不支持右padding

# 模型推理性能

| 硬件形态 | 模型 | Batch | 首token(ms) | 非首token(ms) |
| :-----| ----: | :----: |:----: |:----: |
| 910B3 | LLaMA2-7B | 1 | 46.98 | 15.97

Batch=1, 输入长度和输出长度取[32,64,128,256,512,1024], 共36组case取均值

# 模型推理精度

【基于C-EVAL数据集】

| llama Model 5-shot | Average | Avg(Hard) | STEM  | Social Sciences | Humanities | Others |
| ------------------ | ------- | --------- | ----- | --------------- | ---------- | ------ |
| GPU 7b             | 34.10   | 24.34     | 31.16 | 44.36           | 34.24      | 29.94  |
| NPU 7b (ours)      | 34.47   | 25.0      | 31.63 | 44.36           | 34.6       | 30.47  |

