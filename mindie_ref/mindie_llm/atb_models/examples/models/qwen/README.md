# 特性矩阵

- 此处罗列QWen模型各版本支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16           | Flash Attention | Paged Attention                       | W8A8量化 | W8A16量化 |
|-------------|----------------------------|-----------------------------|----------------|-----------------|---------------------------------------|----------|----------|
| QWen-7B     | √ 支持world size 1,2,4,8   | √ 支持world size 1,2         | √              | √               | √ 支持world size 1,2,4,8               | ×       | ×        |
| Qwen-14B    | √ 支持world size 1,2,4,8   | √ 支持world size 1,2         | √              | √               | √ 支持world size 1,2,4,8               | ×       | ×        |
| QWen-72B    | √ 支持world size 4,8       | ×                            | √ 仅支持800I A2 | √ 仅支持800I A2 | √ 支持world size 1,2,4,8，仅支持800I A2 | ×      | ×         |

# Paged Attention 推理使用说明

## 路径变量解释

| 变量名称     | 含义                                                                                                                                                       |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                                |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models` |
| script_path | 脚本所在路径。QWen系列模型的工作脚本所在路径为${llm_path}/examples/models/qwen                                                                                 |
| weight_path | 模型权重路径                                                                                                                                                |

## 权重转换

Paged Attention 场景需要.safetensors格式的权重，如果没有，参考[此README文件](../../README.md)转换
注：huggingface官网给出的QWen模型权重为.safetensors格式

## 操作说明

在`${llm_path}`目录执行以下指令

```shell
bash bash examples/models/qwen/run_pa.sh -m ${weight_path}
```

注：run_pa.sh中默认RANK_SIZE为8，使用的卡号为0~7，实际使用时可根据需要进行修改。暂不支持奇数卡并行。

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

- 参考[此README文件](../../../tests/modeltest/README.md)

# Flash Attention推理使用说明

| 模型名称  | readme地址                                             |
|----------|--------------------------------------------------------|
| QWen-7B  | [README](../../../pytorch/examples/qwen/7b/README.md)  |
| Qwen-14B | [README](../../../pytorch/examples/qwen/14b/README.md) |
| QWen-72B | [README](../../../pytorch/examples/qwen/72b/README.md) |