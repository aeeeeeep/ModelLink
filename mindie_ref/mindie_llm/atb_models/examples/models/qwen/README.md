# 特性矩阵

- 此处罗列QWen模型各版本支持的特性

| 模型及参数量   | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16         | Flash Attention | Paged Attention                   | W8A8量化 | W8A16量化 |
|----------|----------------------------|-----------------------------|--------------|-----------------|-----------------------------------|--------|---------|
| QWen-7B  | √ 支持world size 1,2,4,8     | √ 支持world size 1,2          | √            | √               | √ 支持world size 1,2,4,8            | ×      | ×       |
| Qwen-14B | √ 支持world size 1,2,4,8     | √ 支持world size 1,2          | √            | √               | √ 支持world size 1,2,4,8            | √      | ×       |
| QWen-72B | √ 支持world size 4,8         | ×                           | √ 仅支持800I A2 | √ 仅支持800I A2    | √ 支持world size 1,2,4,8，仅支持800I A2 | ×      | ×       |

# Paged Attention 推理使用说明

## 路径变量解释

| 变量名称        | 含义                                                                                                                             |
|-------------|--------------------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models` |
| script_path | 脚本所在路径。QWen系列模型的工作脚本所在路径为`${llm_path}/examples/models/qwen`                                                                      |
| weight_path | 模型权重路径                                                                                                                         |

## 权重转换

Paged Attention 场景需要.safetensors格式的权重，如果没有，参考[此README文件](../../README.md)转换
注：huggingface官网给出的QWen模型权重为.safetensors格式

## 推理

### 对话测试

在`${llm_path}`目录执行以下指令

```shell
bash examples/models/qwen/run_pa.sh -m ${weight_path}
```

注：

1.推理支持浮点和量化，若启动浮点推理则在｀${weight_path}｀中传入浮点权重路径，若启动量化则传入量化权重路径（现已支持qwen-14b的w8a8量化推理）

2.执行量化推理时，请在权重路径的config.json文件中添加／修改“quantize”字段，值为相应量化方式，如｀w8a8｀

３.启动qwen需要安装三方依赖tiktoken，若环境中没有该依赖可使用以下命令安装：
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
  bash run.sh pa_fp16 performanc　[［２５６，２５６]) １ qwen ／ｄａｔａ／ｍｏｄｅｌｓ／ｑｗｅｎ－１４ｂ／ ８
  ```

- 参考[此README文件](../../../tests/modeltest/README.md)

# Flash Attention推理使用说明

| 模型名称     | readme地址                                               |
|----------|--------------------------------------------------------|
| QWen-7B  | [README](../../../pytorch/examples/qwen/7b/README.md)  |
| Qwen-14B | [README](../../../pytorch/examples/qwen/14b/README.md) |
| QWen-72B | [README](../../../pytorch/examples/qwen/72b/README.md) |