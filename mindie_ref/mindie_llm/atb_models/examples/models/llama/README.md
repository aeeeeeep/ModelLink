# README

- [LLaMA（Large Language Model Meta AI）](https://github.com/facebookresearch/llama/tree/llama_v1)和 [LLaMA2（Large Language Model Meta AI 2）](https://github.com/facebookresearch/llama)，是由 Meta AI 发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 此代码仓中实现了一套基于NPU硬件的LLaMa推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各LLaMa模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE | TGI |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|--------------|----------|--------|--------|-----|
| LLaMa-7B    | 支持world size 1,2,4,8   | 支持world size 2,4      | 是   | 是   | 是              | 是              | 否       | 否           | 否       | 否     | 是     | 否  |
| LLaMa-13B   | 支持world size 1,2,4,8   | 支持world size 2,4      | 是   | 是   | 是              | 是              | 否       | 否           | 否       | 否     | 是     | 否  |
| LLaMa-33B   | 支持world size 4,8       | 否                      | 是   | 是   | 是              | 是              | 否       | 否           | 否       | 否     | 否     | 否  |
| LLaMa-65B   | 支持world size 8         | 否                      | 是   | 是   | 是              | 是              | 否       | 否           | 否       | 否     | 是     | 否  |
| LLaMa2-7B   | 支持world size 1,2,4,8   | 支持world size 2,4      | 是   | 是   | 是              | 是              | 是       | 否           | 是       | 否     | 是     | 否  |
| LLaMa2-13B  | 支持world size 1,2,4,8   | 支持world size 2,4      | 是   | 是   | 是              | 是              | 是       | 否           | 是       | 否     | 是     | 否  |
| LLaMa2-70B  | 支持world size 8         | 否                      | 是   | 是   | 是              | 是              | 否       | 否           | 否       | 否     | 是     | 否  |

- 此模型仓已适配的模型版本
  - [LLaMa系列](https://github.com/facebookresearch/llama/tree/llama_v1)
  - [LLaMa2系列](https://github.com/facebookresearch/llama/tree/v2)

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models`    |
| script_path | 脚本所在路径；LLaMa和LLaMa2的工作脚本所在路径为${llm_path}/examples/models/llama                            |
| weight_path | 模型权重路径                            |

## 权重转换
- 参考[此README文件](../../README.md)

## 设置通用环境变量
- 将模型仓路径加入Python查询模块和包的搜索路径中
  ```shell
  export PYTHONPATH=${llm_path}:$PYTHONPATH
  ```

## 300I DUO 运行操作说明

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
    - 若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
  - `export MAX_MEMORY_GB=15`
    - 限制最大显存
    - 默认设置最大显存为15GB
    - 若出现显存不足导致的异常，请将该参数改小
  - `export TP_WORLD_SIZE=2`
    - 指定模型运行时的TP数，即world size
    - 默认为单卡双芯
    - 各模型支持的TP数参考“特性矩阵”
    - “单卡双芯”运行请指定`TP_WORLD_SIZE`为`2`，“双卡四芯”运行请指定`TP_WORLD_SIZE`为`4`
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export USE_REFACTOR=true`
    - 是否使用新版模型组图
    - 默认使用
    - 运行llama2-7b和llama2-13b时`use_refactor`参数需设置为False，其余模型运行时需设置为True

## 800I A2 运行操作说明

### 对话测试
**运行Flash Attention FP16**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_800i_a2_fa.sh ${weight_path}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
  - `export MAX_MEMORY_GB=29`
    - 限制最大显存
    - 默认设置最大显存为29GB
    - 若出现显存不足导致的异常，请将该参数改小
  - `export TP_WORLD_SIZE=8`
    - 指定模型运行时的TP数，即world size
    - 默认为八卡
    - 各模型支持的TP数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export USE_REFACTOR=true`
    - 是否使用新版模型组图
    - 默认使用
    - 运行llama2-7b和llama2-13b时`use_refactor`参数需设置为False，其余模型运行时需设置为True
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export HCCL_BUFFSIZE=120
    export HCCL_WHITELIST_DISABLE=1
    export ATB_CONTEXT_WORKSPACE_RING=1
    export ATB_CONTEXT_WORKSPACE_SIZE=2629145600
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=0
    export ATB_LAUNCH_KERNEL_WITH_TILING=0
    export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=1
    export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=0
    ```

**运行Flash Attention BF16**
- 暂不支持

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
  - `export MAX_MEMORY_GB=29`
    - 限制最大显存
    - 默认设置最大显存为29GB
    - 若出现显存不足导致的异常，请将该参数改小
  - `export TP_WORLD_SIZE=8`
    - 指定模型运行时的TP数，即world size
    - 默认为八卡
    - 各模型支持的TP数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export IS_BF16=false`
    - 是否使用BF16精度进行推理
    - 默认使用FP16
  - `export USE_REFACTOR=true`
    - 是否使用新版模型组图
    - 默认使用
    - 运行llama2-7b和llama2-13b时`use_refactor`参数需设置为False，其余模型运行时需设置为True
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export LCCL_ENABLE_FALLBACK=1
    ```

**运行Paged Attention BF16**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
  - 相比于FP16，运行BF16时需修改以下环境变量
    - `export IS_BF16=true`
      - 是否使用BF16精度进行推理
      - 默认使用FP16，运行BF16时需将此环境变量的值设置为true

**运行W8A8量化**
- 待补充

**运行KV cache量化**
- 待补充

**运行稀疏量化**
- 待补充

**运行MOE量化**
- 待补充

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)

## 性能测试
- 进入以下路径
  ```shell
  ${llm_path}/tests/modeltest
  ```
- 运行指令
  ```shell
  bash run.sh pa_fp16 [performance|full_CEval|full_MMLU|full_BoolQ] ([case_pair]) [batch_size] [model_name] ([use_refactor]) [weight_dir] [chip_num] ([max_position_embedding/max_sequence_length])
  ```
    - 参考[此README文件](../../../tests/modeltest/README.md)
  - 运行llama2-7b和llama2-13b时`use_refactor`参数需设置为False，其余模型运行时需设置为True
  - 示例
    ```shell
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama False ${llama2-7b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama False ${llama2-13b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama2-70b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama-7b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama-13b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama-65b权重路径} 8
    ```

## FAQ
- 更多环境变量见`${llm_path}/examples/README.md`
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见`${llm_path}/examples/README.md`
