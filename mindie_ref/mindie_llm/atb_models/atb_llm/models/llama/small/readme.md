# README

llama2-7b/13b服务化解耦框架适配说明

## 权重转换

获取到huggingface原始权重若为.bin格式，需要转为.safetensors

在模型仓根目录执行：

```shell
python -m example.convert.convert_weights --model-path ${path/to/weight}
```

当前量化权重获取之后，也以相同方式对量化权重中的anti_weight进行转换

## 执行命令

在ascend-speed-inference根目录下执行:

```shell
export IS_QUANT=0
export QUANT_WEIGHT_PATH=/path/to/quant_weight/
export QUANT_MODEL_IS_7B=1

torchrun --nproc_per_node ${world_size} --master_port 12347 -m examples.run_pa --model_path ${/path/to/weight} 
```

### 浮点执行

```shell
/path/to/weight 为浮点权重路径
```

### 量化执行

```shell
/path/to/weight 为anti浮点权重路径
```

### 环境变量

```shell
IS_QUANT			量化总开关
QUANT_WEIGHT_PATH	量化权重路径
QUANT_MODEL_IS_7B	回退层选择开关：7b=1,13b=0
```

