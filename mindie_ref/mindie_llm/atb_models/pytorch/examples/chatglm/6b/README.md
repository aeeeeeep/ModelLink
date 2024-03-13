# ChatGLM-6B 模型推理指导 <!-- omit in toc -->

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理前准备](#推理前准备)
- [量化工具使用](#量化工具使用)
- [快速上手](#快速上手)
  - [获取源码及依赖](#获取源码及依赖)
  - [模型推理](#模型推理)
- [模型参考精度和性能结果](#模型参考精度和性能结果)

# 概述

ChatGLM-6B 是一个开源的、支持中英双语问答的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。

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

1. 参见 [推理环境准备](../../../../docs/推理环境准备.md) 安装 固件与驱动，CANN，PyTorchAdapter等基础软件。
   ```shell
   # 使能cann环境变量（根据实际安装路径修改）
   source ${path-to-ascend-toolkit}/set_env.sh
   # 使能加速库环境变量（根据实际安装路径修改）
   source ${path-to-mindie-atb}/set_env.sh
   # 使能inference库环境变量
   source ${path-to-atb-models}/set_env.sh
   ```
   
2. 下载模型实现文件和权重文件，并存储到任意路径下 `CHECKPOINT={path-to-weights}`

     - 推荐下载方式

       ```shell
       # 请自行确认已安装 git-lfs
       git lfs install
       git clone https://huggingface.co/THUDM/chatglm-6b
       cd chatglm-6b
       git reset --hard 2449bdc9d85103734ae987bdc94eafa7fec2145d
       ```

     - 其他下载方式

       如果你的网络环境较差，下载模型参数可能会花费较长时间甚至失败。此时可以先将模型下载到本地，然后从本地加载。
       - 分开下载模型实现文件和权重文件
         ```shell
         # 只下载模型实现文件
         GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b
         cd chatglm-6b
         git reset --hard 2449bdc9d85103734ae987bdc94eafa7fec2145d
         ```
         从 [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b/tree/2449bdc9d85103734ae987bdc94eafa7fec2145d) 手动下载模型权重文件，并将下载的文件替换到本地的 `chatglm-6b` 目录下。

       - 手动从 [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b/tree/2449bdc9d85103734ae987bdc94eafa7fec2145d) 下载所有文件

     - 下载后检查`${CHECKPOINT}`目录如下所示

       ```
       |-- config.json
       |-- configuration_chatglm.py
       |-- modeling_chatglm.py
       |-- pytorch_model-00001-of-00008.bin
       |-- pytorch_model-00002-of-00008.bin
       |-- pytorch_model-00003-of-00008.bin
       |-- pytorch_model-00004-of-00008.bin
       |-- pytorch_model-00005-of-00008.bin
       |-- pytorch_model-00006-of-00008.bin
       |-- pytorch_model-00007-of-00008.bin
       |-- pytorch_model-00008-of-00008.bin
       |-- pytorch_model.bin.index.json
       |-- quantization.py
       |-- tokenization_chatglm.py
       |-- tokenizer_config.json
       `-- ice_text.model
       ```

     - 在config.json中添加如下配置：

       ```
       {
         ......
         "world_size": 1,
         "tie_word_embeddings": false
       }
       ```

4. 下载 `C-Eval` 数据集

   从 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e84444333b6d434ea7b0) 下载处理好的 `C-Eval` 数据集，解压到任意目录下 `DATASET={path-to-dataset}` 。

# 快速上手

## 获取源码及依赖

1. 切换到 `chatglm-6b` 执行目录

   ```shell
   export CHECKPOINT={path-to-weights}
   cd ${path-to-atb-models}/pytorch/examples/chatglm/6b/
   ```
2. 安装第三方依赖

    ```shell
    # torch 1.11.0
    pip3 install torch==1.11.0 torchvision==0.14.1 icetk==0.0.4 transformers==4.30.2 sentencepiece
    
    # torch 2.0.1
    pip3 install torch==2.0.1 torchvision==0.15.2 icetk==0.0.4 transformers==4.30.2 sentencepiece
    ```

## 模型推理

- 可开启CPU Performance模式以提高模型推理性能

  ```shell
  cpupower frequency-set -g performance
  ```
  
- 推理前设置和开启如下环境变量

  ```shell
  export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
  export TASK_QUEUE_ENABLE=1
  export ATB_OPERATION_EXECUTE_ASYNC=1
  export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
  
  # 300 Ipro 和 300 IDuo 上开启
  export HCCL_BUFFSIZE=110
  export ATB_USE_TILING_COPY_STREAM=1
  ```

- `C-Eval` 数据集推理

  ```shell
  # 将TP_SIZE设置为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode precision_dataset --model_path ${CHECKPOINT} --ceval_dataset ${DATASET} --batch 8 --tp_size ${TP_SIZE}
  ```
  
- 模型性能数据测试

  **性能测试请先配置环境变量`export TIMEIT=1`，测试结束后删除该环境变量`unset TIMEIT`。**
  
  ```shell
  # 将TP_SIZE设置为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode performance --model_path ${CHECKPOINT} --batch ${batch_size} --tp_size ${TP_SIZE}
  ```
  
  备注：
  
  1. 可通过配置`--seqlen_in_pair`和`--seqlen_out_pair`指定输入输出序列长度，例如以下命令测试的输入输出组合为[256,256]，[512,512]，[1024,1024]
  
     ```shell
     torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode performance --model_path ${CHECKPOINT} --device 0 --seqlen_in_pair 256,512,1024 --seqlen_out_pair 256,512,1024 --batch ${batch_size} --tp_size ${TP_SIZE} --performance_output_file performance_bs${batch_size}.csv
     ```
  
  2. 环境变量 `MAX_SEQ_LEN` （默认值2048）必须大于等于 `seqlen_in + seqlen_out`，例如：
  
     ```shell
     # 若 seqlen_in = 3584 seqlen_out = 512
     export MAX_SEQ_LEN=4096
     ```
  
- 命令行交互

  ```shell
  # 将TP_SIZE设置为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode cli_demo --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  ```
  
- `main.py` 参数说明：

  ```shell
  # 这里应该是main.py的help信息
  ```

# 模型参考精度和性能结果

待补充
