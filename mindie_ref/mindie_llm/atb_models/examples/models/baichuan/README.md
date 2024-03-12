# 特性矩阵

- 此矩阵罗列了各Baichuan模型支持的特性

| 模型及参数量        | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | Flash Attention | Paged Attention | W8A8量化 |
|---------------|----------------------------|-----------------------------|------|-----------------|-----------------|--------|
| Baichuan2-7B  | 支持world size 1,2,4,8       | 支持world size 2              | 是    | 是               | 是               | 否      |
| Baichuan2-13B | 支持world size 2,4,8         | 支持world size 2,4            | 是    | 是               | 是               | 否      |
| Baichuan-7B   | 支持world size 1,2,4,8       | 支持world size 2              | 是    | 是               | 是               | 否      |
| Baichuan-13B  | 支持world size 2,4,8         | 支持world size 2,4            | 是    | 是               | 是               | 否      |

# Paged Attention 推理使用说明

## 路径变量解释

| 变量名         | 含义                                                                                                                             |
|-------------|--------------------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models` |
| script_path | 脚本所在路径。Baichuan系列模型的工作脚本所在路径为${llm_path}/examples/models/baichuan                                                              |
| weight_path | 模型权重路径                                                                                                                         |

## 权重转换

Paged Attention 场景下需要.safetensors 格式的权重，如果没有，参考[此README文件](../../README.md)转换

## 量化权重转换（W8A8）
- 将当前目录下的convert_w8a8_quant_weights.py文件中的input_fp16_path 和output_w8a8_path 修改为自己的权重路径和输出权重路径
- 如果想用npu转换权重，需要根据注释修改代码将设备设置为npu
- 执行
```
python convert_w8a8_quant_weights.py
```
- 将原权重文件夹下所有json文件拷贝到新的量化权重文件下
- `${weight_path}/config.json`文件中需设置`dtype`和`quantize`类型来标识量化类型和精度
- 若`dtype`和`quantize`字段不存在，需新增

- 配置
  | 量化类型及精度  | torch_dtype | quantize |
  |----------------|-------------|----------|
  | FP16           | "float16"   | ""       |
  | W8A8           | "float16"   | "w8a8"   |

- 示例
  - baichuan模型使用FP16精度，W8A8量化
    ```json
    {
      "torch_dtype": "float16",
      "quantize": "w8a8",
    }
    ```


## 操作说明

### 推理

在`${llm_path}`目录下执行以下指令

```shell
bash examples/models/baichuan/run_pa.sh ${weight_path}
```

根据硬件设备不同请参考下表修改run_pa.sh再运行

### run_pa.sh 参数说明

| 参数名称                      | 含义                    | 800I A2推荐值 | 300I DUO推荐值 |
|---------------------------|-----------------------|------------|-------------|
| BIND_CPU                  | 绑定CPU核心开关,默认进行绑核      | 1          | 1           |
| IS_QUANT                  | 是否启动量化                | 0          | 0           |
| ASCEND_RT_VISIBLE_DEVICES | 使用的硬件卡号，多个卡间使用逗号相连    | 根据实际情况设置   | 根据实际情况设置    |
| RESERVED_MEMORY_GB        | 保留内存，通常未加速库需要的内存+通信内存 | 3          | 3           |
| MASTER_PORT               | 卡间通信端口,通常不用修改，有冲突时再改  |            |             |

## 精度测试

- 参考[此README文件](../../../tests/modeltest/README.md)

## 性能测试

- 参考[此README文件](../../../tests/modeltest/README.md)

# Flash Attention推理使用说明

| 模型名称          | readme地址                                                    |
|---------------|-------------------------------------------------------------|
| Baichuan2-7B  | [README](../../../pytorch/examples/baichuan2/7b/README.md)  |
| Baichuan2-13B | [README](../../../pytorch/examples/baichuan2/13b/README.md) |
| Baichuan-7B   | [README](../../../pytorch/examples/baichuan/7b/README.md)   |
| Baichuan-13B  | [README](../../../pytorch/examples/baichuan/13b/README.md)  |
