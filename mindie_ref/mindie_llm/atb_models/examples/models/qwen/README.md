# 特性矩阵

- 此处罗列QWen模型各版本支持的特性

| 模型及参数量   | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16         | Flash Attention | Paged Attention                   | W8A8量化 | W8A16量化 |
|----------|----------------------------|-----------------------------|--------------|-----------------|-----------------------------------|--------|---------|
| QWen-7B  | √ 支持world size 1,2,4,8     | √ 支持world size 1,2          | √            | √               | √ 支持world size 1,2,4,8            | ×      | ×       |
| Qwen-14B | √ 支持world size 1,2,4,8     | √ 支持world size 1,2          | √            | √               | √ 支持world size 1,2,4,8            | √      | ×       |
| QWen-72B | √ 支持world size 4,8         | ×                           | √ 仅支持800I A2 | √ 仅支持800I A2    | √ 支持world size 1,2,4,8，仅支持800I A2 | ×      | √       |

# Paged Attention 推理使用说明

注意：
- 模型权重所在路径中的config.json文件需添加字段`torch_dtype`，例如`"torch_dtype": "float16"`
- 执行量化推理时，须在量化权重所在路径的config.json文件中添加字段`quantize`，值为当前量化权重的量化方式，例如`"quantize": "w8a8"`、`"quantize": "w8a16"`

## 路径变量解释

| 变量名称        | 含义                                                                                                                             |
|-------------|--------------------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models` |
| script_path | 脚本所在路径。QWen系列模型的工作脚本所在路径为`${llm_path}/examples/models/qwen`                                                                      |
| weight_path | 模型权重路径                                                                                                                         |

## 权重格式转换

Paged Attention 场景需要.safetensors格式的权重，如果没有，参考[此README文件](../../README.md)转换
注：huggingface官网给出的QWen模型权重为.safetensors格式

## 量化权重导出
量化权重可通过ModelSlim（昇腾压缩加速工具）实现。

#### 环境准备
环境配置可参考ModelSlim官网：https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/devtools/auxiliarydevtool/modelslim_0002.html

#### 导出量化权重
通过`${llm_path}/examples/models/qwen/quant_qwen_14b_w8a8.py`和`${llm_path}/examples/models/qwen/quant_qwen_72b_w8a16.py`文件导出目标模型的量化权重（注意量化权重不要和浮点权重放在同一个目录下）：
```shell
python quant_qwen_14b_w8a8.py ${浮点权重路径} ${量化权重保存路径}
```
导出量化权重后应生成`quant_model_weight.safetensors`和`quant_model_description.json`两个文件。

注：

1.quant_qwen_14b_w8a8.py和quant_qwen_72b_w8a16.py文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.启动量化推理时，需要将config.json等相关文件复制到量化权重路径中，可执行以下指令进行复制：
```shell
cp ${浮点权重路径}/*.py ${量化权重路径}
cp ${浮点权重路径}/*.json ${量化权重路径}
cp ${浮点权重路径}/*.tiktoken ${量化权重路径}
```

3.启动量化推理时，请在权重路径的config.json文件中添加(或修改)`quantize`字段，值为相应量化方式，如`w8a8`或`w8a16`。

## 推理

### 对话测试

在`${llm_path}`目录执行以下指令

```shell
bash examples/models/qwen/run_pa.sh -m ${weight_path}
```

注：

1.推理支持浮点和量化，若启动浮点推理则在`${weight_path}`中传入浮点权重路径，若启动量化则传入量化权重路径

2.启动qwen需要安装三方依赖tiktoken，若环境中没有该依赖可使用以下命令安装：
```shell
pip install tiktoken
```

根据硬件设备不同请参考下表修改run_pa.sh再运行

### run_pa.sh 参数说明

| 参数名称                      | 含义                    | 800I A2推荐值 | 300I DUO推荐值 |
|---------------------------|-----------------------|------------|-------------|
| BIND_CPU                  | 绑定CPU核心开关,默认进行绑核      | 1          | 1           |
| ASCEND_RT_VISIBLE_DEVICES | 使用的硬件卡号，多个卡间使用逗号相连    | 根据实际情况设置   | 根据实际情况设置    |
| RESERVED_MEMORY_GB        | 保留内存，通常未加速库需要的内存+通信内存 | 3          | 3           |
| MASTER_PORT               | 卡间通信端口,通常不用修改，有冲突时再改  |            |             |

注：暂不支持奇数卡并行

## 精度测试

- 参考[此README文件](../../../tests/modeltest/README.md)

## 性能测试

- 进入以下路径
  ```shell
  ${llm_path}/tests/modeltest
  ```
- 运行指令
  ```shell
  bash run.sh pa_fp16 [performance|full_CEval|full_BoolQ] ([case_pair]) [batch_size] qwen ([use_refactor]) [weight_dir] [chip_num] ([max_position_embedding/max_sequence_length])
  ```

示例：
  ```shell
  bash run.sh pa_fp16 performance　[[256,256]] 1 qwen /data/models/qwen-14b/ 8
  ```

- 参考[此README文件](../../../tests/modeltest/README.md)

# Flash Attention推理使用说明

| 模型名称     | readme地址                                               |
|----------|--------------------------------------------------------|
| QWen-7B  | [README](../../../pytorch/examples/qwen/7b/README.md)  |
| Qwen-14B | [README](../../../pytorch/examples/qwen/14b/README.md) |
| QWen-72B | [README](../../../pytorch/examples/qwen/72b/README.md) |