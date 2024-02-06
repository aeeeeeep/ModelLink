# CodeGeeX2-6B 模型推理指导 <!-- omit in toc -->

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理前准备](#推理前准备)
- [生成量化权重](#生成量化权重)
- [快速上手](#快速上手)
  - [获取源码及依赖](#获取源码及依赖)
  - [模型推理](#模型推理)
- [模型参考精度和性能结果](#模型参考精度和性能结果)

# 概述

[CodeGeeX2-6B](https://github.com/THUDM/CodeGeeX2) 是多语言代码生成模型 [CodeGeeX](https://github.com/THUDM/CodeGeeX) ([KDD’23](https://arxiv.org/abs/2303.17568)) 的第二代模型。不同于一代 CodeGeeX（完全在国产华为昇腾芯片平台训练） ，CodeGeeX2 是基于 [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) 架构加入代码预训练实现，得益于 ChatGLM2 的更优性能，CodeGeeX2 在多项指标上取得性能提升（+107% > CodeGeeX；仅60亿参数即超过150亿参数的 StarCoder-15B 近10%）。

# 输入输出数据

- 输入数据

  | 输入数据       | 大小                 | 数据类型 | 数据排布格式 | 是否必选 |
  | -------------- | -------------------- | -------- | ------------ | -------- |
  | input_ids      | BATCH_SIZE x SEQ_LEN | INT64    | ND           | 是       |
  | attention_mask | BATCH_SIZE x SEQ_LEN | FLOAT32  | ND           | 否       |

- 输出数据

  | 输出数据   | 大小                        | 数据类型 | 数据排布格式 |
  | ---------- | --------------------------- | -------- | ------------ |
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64    | ND           |

# 推理前准备

1. 参见 [推理环境准备](../../../docs/推理环境准备.md) 安装 固件与驱动，CANN，PyTorchAdapter等基础软件。
   ```shell
   # 使能cann环境变量（根据实际安装路径修改）
   source ${path-to-ascend-toolkit}/set_env.sh
   # 使能加速库环境变量（根据实际安装路径修改）
   source ${path-to-ascendTB}/set_env.sh
   # 使能inference库环境变量
   source ${path-to-transfomer-llm}/set_env.sh
   #稀疏工具在线编译
   cd ${path-to-ascend-toolkit}/tools/modelslim/pytorch/weight_compression/compress_graph/
   bash build.sh ${path-to-ascend-toolkit}/ascend-toolkit/latest/
   ```

2. 下载模型实现文件和权重文件，并存储到任意路径下 `CHECKPOINT={path-to-weights}`

     - 推荐下载方式

       ```shell
       # 请自行确认已安装 git-lfs
       git lfs install
       git clone https://huggingface.co/THUDM/codegeex2-6b
       ```

     - 其他下载方式

       如果你的网络环境较差，下载模型参数可能会花费较长时间甚至失败。此时可以先将模型下载到本地，然后从本地加载。

       - 手动从 [THUDM/codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b) 下载所有文件

     - 下载后检查`${CHECKPOINT}`目录如下所示

       ```
       |-- config.json
       |-- configuration_chatglm.py
       |-- generation_config.json
       |-- modeling_chatglm.py
       |-- pytorch_model-00001-of-00007.bin
       |-- pytorch_model-00002-of-00007.bin
       |-- pytorch_model-00003-of-00007.bin
       |-- pytorch_model-00004-of-00007.bin
       |-- pytorch_model-00005-of-00007.bin
       |-- pytorch_model-00006-of-00007.bin
       |-- pytorch_model-00007-of-00007.bin
       |-- pytorch_model.bin.index.json
       |-- quantization.py
       |-- save_model.py
       |-- tokenization_chatglm.py
       |-- tokenizer_config.json
       |-- tokenizer.model
       ```

     - 在config.json中添加如下配置：

       ```
       {
         ......
         "world_size": 1,
         "float_layers_id": [0]
       }
       ```

3. 准备`HumanEval-X`数据集，将CodeGeeX2源码仓下载到当前目录下

    ```
    git clone https://github.com/THUDM/CodeGeeX2
    cd CodeGeeX2
    git apply ../dataset.diff
    cd ..
    ```

# 生成量化权重

参见[ChatGLM2 量化工具使用](../../chatglm2/6b/README.md#量化工具使用)，直接使用ChatGLM2文件生成CodeGeeX2量化权重（将`${CHECKPOINT}`改为CodeGeeX2的权重路径）。

# 快速上手

## 获取源码及依赖

1. 获取源码

   ```shell
   cd ${path-to-transfomer-llm}/pytorch/examples/codegeex2/6b
   ```
2. 安装第三方依赖

    ```shell
    pip install -r requirements.txt
    ```

## 模型推理

- 可开启CPU Performance模式以提高模型推理性能

    ```
    cpupower frequency-set -g performance
    ```

- 推理前开启如下环境变量

    ```shell
    export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
    export TASK_QUEUE_ENABLE=1
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1

    # 仅300 Ipro和300 IDuo上开启
    export HCCL_BUFFSIZE=110
    export ATB_USE_TILING_COPY_STREAM=1
    ```

- `HumanEval-X`数据集推理

    ```
    cp ../../chatglm2/6b/patches/models/modeling_chatglm_ascend.py 
    cd CodeGeeX2
    bash script/run_humanevalx.sh
    ```

- 模型性能数据测试

    参见[ChatGLM2 模型推理-模型性能数据测试](../../chatglm2/6b/README.md#模型推理)，直接使用ChatGLM2文件进行测试（将`${CHECKPOINT}`改为CodeGeeX2的权重路径）。

- UI 交互

  参见[ChatGLM2 模型推理-UI 交互](../../chatglm2/6b/README.md#模型推理)，直接使用ChatGLM2文件进行测试（将`${CHECKPOINT}`改为CodeGeeX2的权重路径）。

# 模型参考精度和性能结果

- 参考精度

  > 因为 `C-Eval` 数据集test子集需要上传官网得到结果，所以这里使用val子集进行精度对比

  | ChatGLM2   | 类别 | Average Accuracy |
  | ---------- | ---- | ---------------- |
  | GPU (浮点bs1)  | Python(Pass@1)  | 35.9%           |
  | NPU (浮点bs1)  | Python(Pass@1)  | 35.9%           |

- 推理性能

  > 这里性能结果仅作为参考，并非版本极致性能优化结果。

  | 硬件形态 | 批大小 | 输入长度 | 输出长度 | 解码速度 |
  | -------- | ------ | -------- | -------- | -------- |
  | 300I Duo | 1      | 8192     | 1024     | 162ms    |
