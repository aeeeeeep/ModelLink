# README

- [LLaMA（Large Language Model Meta AI）](https://github.com/facebookresearch/llama/tree/llama_v1)和 [LLaMA2（Large Language Model Meta AI 2）](https://github.com/facebookresearch/llama)，是由 Meta AI 发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 此代码仓中实现了一套基于NPU硬件的LLaMa推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各LLaMa模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE | TGI |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|---------|--------------|----------|--------|--------|-----|
| LLaMa-7B    | 支持world size 1,2,4,8   | 支持world size 2,4      | 是   | 是   | 是              | 是              | 否       | 否       | 否           | 否       | 否     | 是     | 否  |
| LLaMa-13B   | 支持world size 1,2,4,8   | 支持world size 2,4      | 是   | 是   | 是              | 是              | 否       | 否       | 否           | 否       | 否     | 是     | 否  |
| LLaMa-33B   | 支持world size 4,8       | 否                      | 是   | 是   | 是              | 是              | 否       | 否       | 否           | 否       | 否     | 否     | 否  |
| LLaMa-65B   | 支持world size 8         | 否                      | 是   | 是   | 是              | 是              | 否       | 是       | 否           | 否       | 否     | 是     | 否  |
| LLaMa2-7B   | 支持world size 1,2,4,8   | 支持world size 2,4      | 是   | 是   | 是              | 是              | 是       | 否       | 否           | 否       | 否     | 是     | 否  |
| LLaMa2-13B  | 支持world size 1,2,4,8   | 支持world size 2,4      | 是   | 是   | 是              | 是              | 是       | 否       | 否           | 否       | 否     | 是     | 否  |
| LLaMa2-70B  | 支持world size 8         | 否                      | 是   | 是   | 是              | 是              | 是       | 是      | 否       | 否       | 否     | 是     | 否  |

- 此模型仓已适配的模型版本
  - [LLaMa系列](https://github.com/facebookresearch/llama/tree/llama_v1)
  - [LLaMa2系列](https://github.com/facebookresearch/llama/tree/v2)

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models`    |
| script_path | 脚本所在路径；LLaMa和LLaMa2的工作脚本所在路径为`${llm_path}/examples/models/llama`                            |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**
- [LLaMa-7B](https://huggingface.co/huggyllama/llama-7b)
- [LLaMa-13B](https://huggingface.co/huggyllama/llama-13b)
- [LLaMa-33B](https://huggingface.co/pinkmanlove/llama-33b-hf/tree/main)
- [LLaMa-65B](https://huggingface.co/huggyllama/llama-65b)
- [LLaMa2-7B](https://huggingface.co/NousResearch/Llama-2-7b-hf)
- [LLaMa2-13B](https://huggingface.co/NousResearch/Llama-2-13b-hf)
- [LLaMa2-70B](https://huggingface.co/NousResearch/Llama-2-70b-hf)

**权重转换**
- 参考[此README文件](../../README.md)

**量化权重生成**
- 基于原始的FP16的权重，生成量化权重
- W8A8量化权重请使用以下指令生成
  - 设置环境变量
  ```shell
  # 设置CANN包的环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  # 推荐使用transformers 4.36.2版本进行权重转换，但执行模型推理时transformers的版本仍需为4.30.2
  pip uninstall transformers
  pip install transformers=={指定版本}
  ```
  - 打开量化脚本convert_w8a8_quant_weights.py，配置量化参数
  ```shell
  IN_MODEL_PATH = './llama2-7b' # 浮点权重输入路径
  OUT_MODEL_PATH = './llama2-7b_quant' #量化权重生成路径
  NUM_LAYERS = 32 # 模型层数，LLaMA2-7B配置为32，13B配置为40
  ANTI_METHOD = "m1" # anti-outlier算法配置，LLaMA2-7B配置为m1，13B配置为m2
  ```
  - 打开量化脚本convert_w8a8_quant_weights.py，配置校准数据集
  ```shell
  # 跳转到脚本第33行，从 https://github.com/google-research-datasets/boolean-questions 下载BoolQ Development数据集，从数据集中选取50条问题，填入calib_list中
  calib_list = [
    # 选取50条数据填入
    "Question1: xxx",
    "Question2: xxx",
  ]
  ```
  - 执行量化脚本
  ```shell
  python convert_w8a8_quant_weights.py 
  ```
- W8A16量化权重请使用以下指令生成
  - 当前仅LLaMa-65B和LLaMa2-70B支持W8A16量化
  ```shell
  # 设置CANN包的环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd ${llm_path}
  python examples/models/llama/convert_w8a16_quant_weights.py --fp16_model_path {浮点权重路径} --w8a16_model_path {W8A16量化权重路径}
  ```
    - 注意：`fp16_model_path`和`w8a16_model_path`请勿使用同一个文件夹，避免浮点权重和量化权重混淆
  - 示例
    ```shell
    python examples/models/llama/convert_w8a16_quant_weights.py --fp16_model_path /home/weights/llama2-70b --w8a16_model_path /home/weights/llama2-70b_w8a16
    ```
  - 推荐使用transformers 4.36.2版本进行权重转换，transformers 4.36.2版本会大大加快权重生成的速度，但执行模型推理时transformers的版本仍需为4.30.2
    ```shell
    # 卸载
    pip uninstall transformers
    # 安装
    pip install transformers=={指定版本}
    ```

**LLaMa 33B权重添加Special token**
- LLaMa 33B中tokenizer原始的special token为空，需手动将权重文件中的`special_tokens_map.json`文件替换成以下内容
  ```json
  {
    "add_bos_token": true,
    "add_eos_token": false,
    "bos_token": {
      "content": "<s>",
      "lstrip": false,
      "normalized": true,
      "rstrip": false,
      "single_word": false
    },
    "clean_up_tokenization_spaces": false,
    "eos_token": {
      "content": "</s>",
      "lstrip": false,
      "normalized": true,
      "rstrip": false,
      "single_word": false
    },
    "model_max_length": 2048,
    "pad_token": null,
    "sp_model_kwargs": {},
    "tokenizer_class": "LlamaTokenizer",
    "unk_token": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": true,
      "rstrip": false,
      "single_word": false
    }
  }
  ```


**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试
**运行Flash Attention FP16**
- 其余LLaMa模型参考以下运行方式
  - 运行启动脚本
    - 在\${llm_path}目录下执行以下指令
      ```shell
      bash ${script_path}/run_fa.sh ${weight_path}
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

**运行Flash Attention W8A8**
- 运行启动脚本
  - 与“运行Flash Attention FP16”的启动方式相同
  - `${weight_path}`为W8A8量化权重的路径
- 环境变量说明
  - 参见“运行Flash Attention FP16”中的环境变量说明
- 相比于FP16，运行量化时需修改W8A8量化权重`${weight_path}/config.json`中的`quantize`字段，将此字段对应的值修改为`w8a8`
  - 若config.json中无此字段，则新增

**运行Flash Attention W8A16**
- 运行启动脚本
  - 与“运行Flash Attention FP16”的启动方式相同
  - `${weight_path}`为W8A16量化权重的路径
- 环境变量说明
  - 参见“运行Flash Attention FP16”中的环境变量说明

**运行Paged Attention FP16**
- 运行启动脚本
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

**运行Paged Attention BF16**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
- 相比于FP16，运行BF16时需修改${weight_path}/config.json中的`torch_dtype`字段，将此字段对应的值修改为`bfloat16`
- 300I DUO卡暂不支持BF16特性

**运行Paged Attention W8A8**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
  - `${weight_path}`为W8A8量化权重的路径
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
- 相比于FP16，运行量化时需修改W8A8量化权重`${weight_path}/config.json`中的`quantize`字段，将此字段对应的值修改为`w8a8`
  - 若config.json中无此字段，则新增

**运行Paged Attention W8A16**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
  - `${weight_path}`为W8A16量化权重的路径
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明

**运行KV cache量化**
- 待补充

**运行稀疏量化**
- 待补充

**运行MOE量化**
- 待补充

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export MAX_MEMORY_GB=29
    bash run.sh pa_fp16 full_BoolQ 1 llama True ${llama2-7b权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 llama True ${llama2-13b权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 llama True ${llama2-70b权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 llama True ${llama-7b权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 llama True ${llama-13b权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 llama True ${llama-65b权重路径} 8
    ```
- 运行量化权重和BF16时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](../../README.md)

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export MAX_MEMORY_GB=29
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama2-7b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama2-13b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama2-70b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama-7b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama-13b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama True ${llama-65b权重路径} 8
    ```
- 运行量化权重和BF16时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](../../README.md)

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)
- 运行时，需要通过指令pip list｜grep protobuf确认protobuf版本，如果版本高于3.20.x，请运行指令pip install protobuf==3.20.0进行更新
