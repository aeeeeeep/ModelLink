# ChatGLM2-6B 模型推理指导 <!-- omit in toc -->

# 概述

- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/) 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B有更强大的性能、更长的上下文、更高效的推理和更开放的协议。
- 此代码仓中实现了一套基于NPU硬件的ChatGLM2推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了ChatGLM2-6B模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE | TGI |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|--------------|----------|--------|--------|-----|
| ChatGLM2-6B    | 支持world size 1,2  | 支持world size 1,2      | 是   | 否   | 否              | 是              | 否       | 否           | 否       | 否     | 是     | 是  |

- 此模型仓已适配的模型版本
  - [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b/tree/main)

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models`    |
| script_path | 脚本所在路径；路径为${llm_path}/examples/models/chatglm/v2_6b                            |
| weight_path | 模型权重路径                            |

## 权重转换
- 参考[此README文件](../../../README.md)

## 量化权重导出
量化权重可通过ModelSlim（昇腾压缩加速工具）实现。

#### 环境准备
环境配置可参考ModelSlim官网：https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/devtools/auxiliarydevtool/modelslim_0002.html

#### 导出量化权重
通过`${llm_path}/examples/models/chatglm/v2_6b/quant_chatglm2_6b_w8a8.py`文件导出模型的量化权重（注意量化权重不要和浮点权重放在同一个目录下）：
```shell
python quant_chatglm2_6b_w8a8.py --model_path ${浮点权重路径} --save_path ${量化权重保存路径}
```
导出量化权重后应生成`quant_model_weight_w8a8.safetensors`和`quant_model_description_w8a8.json`两个文件。

注：

1.quant_chatglm2_6b_w8a8.py文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.启动量化推理时，需要将config.json等相关文件复制到量化权重路径中，可执行以下指令进行复制：
```shell
cp ${浮点权重路径}/config* ${量化权重路径}
cp ${浮点权重路径}/tokeniz* ${量化权重路径}
cp ${浮点权重路径}/modeling* ${量化权重路径}
```

3.启动量化推理时，请在权重路径的config.json文件中添加(或修改)`quantize`字段，值为相应量化方式，当前仅支持`w8a8`。

4.执行完以上步骤后，执行量化模型只需要替换权重路径。

## 300I DUO 运行操作说明
- 可开启CPU Performance模式以提高模型推理性能

  ```
  cpupower frequency-set -g performance
  ```

### 对话测试
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_300i_duo.sh ${weight_path}
    ```
- 环境变量说明
  - `export BIND_CPU=1`
    - 绑定CPU核心开关
    - 默认进行绑核
    - 若当前机器未设置NUMA或绑核失败，可将 BIND_CPU 设为 0
  - `export IS_QUANT=0`
    - 量化开关
    - 默认非量化
    - 若需要开启请参照[文档](../../../atb_llm/models/llama/small/readme.md)
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
  - `export TP_WORLD_SIZE=2`
    - 指定模型运行时的TP数，即world size
    - 默认为单卡双芯
    - 各模型支持的TP数参考“特性矩阵”
    - “单卡双芯”运行请指定`TP_WORLD_SIZE`为`2`
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入Python查询模块和包的搜索路径中
    - 将${llm_path}替换为实际路径
  - - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    # 内存
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    # 性能
    export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=110
    ```


## 800I A2 运行操作说明
- 可开启CPU Performance模式以提高模型推理性能

  ```
  cpupower frequency-set -g performance
  ```
### 对话测试
**运行Flash Attention FP16**
- 请查看[此README文件](../../../../pytorch/examples/chatglm2/6b/README.md)


**运行Paged Attention FP16**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_800i_a2_pa.sh ${weight_path}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
  - `export TP_WORLD_SIZE=1`
    - 指定模型运行时的TP数，即world size
    - 默认为单卡
    - 各模型支持的TP数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入Python查询模块和包的搜索路径中
    - 将${llm_path}替换为实际路径
  - `export IS_BF16=false`
    - 是否使用BF16精度进行推理
    - 默认使用FP16
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    # 内存
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    # 性能
    export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export LCCL_ENABLE_FALLBACK=1
    ```


**运行Paged Attention BF16**
- 暂不支持

**运行W8A8量化**
- 暂不支持

**运行KV cache量化**
- 暂不支持

**运行稀疏量化**
- 暂不支持

**运行MOE量化**
- 暂不支持


## 精度测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## FAQ
- `import torch_npu`遇到`xxx/libgomp.so.1: cannot allocate memory in static TLS block`报错，可通过配置`LD_PRELOAD`解决。
  - 示例：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`