# STARCODER README

StarCoder模型是在The Stack (v1.2)的80+种编程语言上训练的15.5B参数模型，不包括选择退出请求。该模型使用多查询注意力，一个包含8192个令牌的上下文窗口，并在1万亿个令牌上使用填充中间目标进行训练。

- 参考实现：
```
https://huggingface.co/bigcode/starcoder
```

# 使用说明

##权重

- 下载starcoder模型权重，放置到自定义`model_path`
```
https://huggingface.co/bigcode/starcoder/tree/main
```
- 进入刚才下载的权重文件夹中将config.json文件中的 "model_type": "gpt_bigcode" 修改为 "model_type": "starcoder" 
- 使用`/path-to-ModelLink/mindie_ref/mindie_llm/atb_models/examples/convert/convert_weights.py`将bin转成safetensor格式
- 示例
```shell
python /path-to-ModelLink/mindie_ref/mindie_llm/atb_models/examples/convert/convert_weights.py --model_path {bin文件权重的路径}
```
- 输出结果会保存在bin权重同目录下

## 310P 运行操作说明

- 设置环境变量
```shell
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_LLM_BENCHMARK_ENABLE=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_LAUNCH_KERNEL_WITH_TILING=1
```
### 其余操作同800I A2

## 800I A2 运行操作说明

### 环境变量
```shell
# source cann环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source 加速库环境变量
source /usr/local/Ascend/atb/set_env.sh
# source 模型仓tar包解压出来后的环境变量
source set_env.sh
```

### 安装python依赖
```
pip install loguru
pip install tabulate
```

### 参数说明
- `--model_path` 模型路径
- `--input_text` 输入问题
- `--input_file` 已文件形式批量传入输入问题，输入需经过tokenizer转换为token id
- `--max_input_length` 最大输入长度
- `--max_output_length` 最大输出长度
- `--max_batch_size` 每次运行时固定的batch数量
- `--is_flash_causal_lm`
    - Flash Attention时应设为False，Paged Attention时应设为True
    - `run_fa.py`和`run_pa.py`脚本已自动做过适配，无需手动传入
- `--is_bf16`
    - 默认精度为`FP16`，若运行时传入此参数，则精度设置为`BF16`
    - 注意：当前仅Paged Attention支持打开此开关

- 所有参数可见run_pa.py文件中

### 对话测试
**运行Flash Attention FP16**
- 暂不支持

**运行Flash Attention BF16**
- 暂不支持

**运行Paged Attention FP16**

- 设置环境变量
```shell
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=1
export ATB_LLM_BENCHMARK_ENABLE=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export LCCL_ENABLE_FALLBACK=1
```

- 运行指令（在/path-to-ModelLink/mindie_ref/mindie_llm/atb_models/路径下运行以上指令）
```shell
torchrun --nproc_per_node {TP数，即world size} --master-port {卡间通信端口} -m examples.run_pa --model_path {模型的权重路径}
```
- 示例
```shell
torchrun --nproc_per_node 8 --master_port 12345 -m examples.run_pa --model_path /path/to/model --input_text "def print_hello_word()" --max_input_length 20 --max_output_length 50 --max_batch_size 1
```

**运行Paged Attention BF16**    
- 待补充(同FP16)

**运行W8A8量化**
- 权重转换

**运行KV cache量化**
- 待补充

**运行稀疏量化**
- 待补充

**运行MOE量化**
- 待补充

## 精度测试
- 待补充

## 性能测试
- 待补充

## 性能数据
- 待补充

