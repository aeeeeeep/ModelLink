# README

[Yi系列模型](https://huggingface.co/01-ai) 是由 01.AI 从头开始训练的新一代开源大型语言模型。[Yi系列模型](https://huggingface.co/01-ai) 以双语语言模型为目标，在 3T 多语种语料库上进行训练，已成为全球最强大的 LLM 之一，在语言理解、常识推理、阅读理解等方面展示出良好的前景。

- 此代码仓中实现了一套基于NPU硬件的Yi系列模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各Yi模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE | TGI |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|---------|--------------|----------|--------|--------|-----|
| Yi-6B-200K    | 支持world size 1,2,4,8   | 否     | 是   | 是   | 否              | 是              | 否       | 否       | 否           | 否       | 否     | 否     | 否  |
| Yi-34B    | 支持world size 1,2,4,8   | 否     | 是   | 是   | 否              | 是              | 否       | 否       | 否           | 否       | 否     | 否     | 否  |

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models`    |
| script_path | 脚本所在路径；Yi-6B-200K和Yi-34B的工作脚本所在路径为`${llm_path}/examples/models/yi`                            |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**

- [Yi-6B-200K](https://huggingface.co/01-ai/Yi-6B-200K)
- [Yi-34B](https://huggingface.co/01-ai/Yi-34B)

**权重转换**
- 参考[此README文件](../../README.md)

**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试

**运行Paged Attention BF16**
- 运行启动脚本
  - 将`${llm_path}`加入`PYTHONPATH`搜索目录
    ```shell
    export PYTHONPATH=${llm_path}:${PYTHONPATH}
    ```
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
    - 各模型支持的核心数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export USE_REFACTOR=true`
    - 是否使用新版模型组图
    - 默认使用
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export LCCL_ENABLE_FALLBACK=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

**运行Paged Attention FP16**
- 运行启动脚本
  - 与“运行Paged Attention BF16”的启动方式相同
- 环境变量说明
  - 参见“运行Paged Attention BF16”中的环境变量说明
- 相比于BF16，运行FP16时需修改${weight_path}/config.json中的`torch_dtype`字段，将此字段对应的值修改为`float16`

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export MAX_MEMORY_GB=29
    bash run.sh pa_bf16 full_CEval 1 llama True ${Yi-6B-200K权重路径} 8
    bash run.sh pa_bf16 full_CEval 1 llama True ${Yi-34B权重路径} 8
    ```

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export MAX_MEMORY_GB=29
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${Yi-6B-200K权重路径} 8
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${Yi-34B权重路径} 8
    ```

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)
- 运行时，需要通过指令pip list｜grep protobuf确认protobuf版本，如果版本高于3.20.x，请运行指令pip install protobuf==3.20.0进行更新
