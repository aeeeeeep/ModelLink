# STARCODER README

StarCoder模型是在The Stack (v1.2)的80+种编程语言上训练的15.5B参数模型，不包括选择退出请求。该模型使用多查询注意力，一个包含8192个令牌的上下文窗口，并在1万亿个令牌上使用填充中间目标进行训练。

- 参考实现：
```
https://huggingface.co/bigcode/starcoder
```

# 使用说明

## 权重下载
- 下载starcoder模型权重，放置到自定义路径下
```
https://huggingface.co/bigcode/starcoder/tree/main
```
- 进入刚才下载的权重文件夹中将config.json文件中的 "model_type": "gpt_bigcode" 修改为 "model_type": "starcoder" 
- 权重文件夹中将config.json文件中的 "torch_type": "float32" 修改为 "torch_type": "float16" 

## 权重转换
- 参考[此README文件](../../README.md)

## 量化权重转换（W8A8）
- 转换量化权重时需先暂时将原权重下config.json中的的"model_type": "starcoder" 修改回 "model_type": "gpt_bigcode" （后续运行时需改回starcoder）
- 将当前目录下的convert_w8a8_quant_weights.py文件中的第29行和30行修改为自己的权重路径，将67行修改为输出权重路径
- 去目标文件目录下执行(其中有两种配置，性能测试请用配置1，精度测试请用配置2，注释掉其中一种配置来执行另外一个配置)
```
python convert_w8a8_quant_weights.py
```
- 将原权重文件夹下所有json文件拷贝到新的量化权重文件下
- 将生成的quant_model_description.json文件名修改为quant_model_description_w8a8.json
- `${weight_path}/config.json`文件中需设置`dtype`和`quantize`类型来标识量化类型和精度
- 若`dtype`和`quantize`字段不存在，需新增

- 配置
  | 量化类型及精度  | torch_dtype | quantize |
  |----------------|-------------|----------|
  | FP16           | "float16"   | ""       |
  | BF16           | "bfloat16"  | ""       |
  | W8A8           | "float16"   | "w8a8"   |
  | W8A16          | "float16"   | "w8a16"  |

- 示例
  - starcoder模型使用FP16精度，W8A8量化
    ```json
    {
      "torch_dtype": "float16",
      "quantize": "w8a8",
    }
    ```

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models`    |
| script_path | 脚本所在路径；starcoder的工作脚本所在路径为${llm_path}/examples/models/starcoder                          |
| weight_path | 模型权重路径 |

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
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
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
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入Python查询模块和包的搜索路径中
    - 将${llm_path}替换为实际路径

### 对话测试脚本参数说明
- `--model_path` 模型路径
- `--input_text` 输入问题
- `--max_input_length` 最大输入长度
- `--max_output_length` 最大输出长度
- `--max_batch_size` 每次运行时固定的batch数量
- 所有参数可见run_pa.py文件中

## 800I A2 运行操作说明

### 对话测试
**运行Flash Attention FP16**
- 暂不支持

**运行Flash Attention BF16**
- 暂不支持

**运行Paged Attention FP16**

### 对话测试
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_800i_a2_pa.sh ${weight_path}
    ```
- 环境变量说明
  - `export BIND_CPU=1`
    - 绑定CPU核心开关
    - 默认进行绑核
    - 若当前机器未设置NUMA或绑核失败，可将 BIND_CPU 设为 0
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
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
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入Python查询模块和包的搜索路径中
    - 将${llm_path}替换为实际路径

### 对话测试脚本参数说明
- `--model_path` 模型路径
- `--input_text` 输入问题
- `--max_input_length` 最大输入长度
- `--max_output_length` 最大输出长度
- `--max_batch_size` 每次运行时固定的batch数量
- 所有参数可见run_pa.py文件中

**运行Paged Attention BF16**    
- 待补充

**运行W8A8量化**
- 获取量化权重后操作步骤同上

**运行KV cache量化**
- 待补充

**运行稀疏量化**
- 待补充

**运行MOE量化**
- 待补充

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)



