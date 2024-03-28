# BLOOM 176B

| 模型                         | FP16      | W8A16     |
| ---------------------------- | --------- | --------- |
| BLOOM-176B (Flash Attention) | 支持910B3 | 支持910B4 |
| BLOOM-176B (Paged Attention) | 未测试    | 支持910B4 |

## 环境配置

### CANN

请使用 CANN-8.0.RC1.B030 或更高的版本。

### Python

建议使用 Python 3.9.18

### PyTorch NPU

建议使用 torch 2.0.1，torch_npu 2.0.1.post1-20240125

### 其它依赖

```shell
pip install transformers==4.34.0
pip install llmtask==0.0.2
```

## 运行测试

运行测试前，请切换到符合上述依赖要求的环境，并：1、 source CANN 包；2、source 加速库；3、source 模型仓。

### 权重准备

下面涉及到的本地路径的描述都使用方括号【PATH】来表示，实际操作时请用本地路径替换掉【PATH】。

#### W8A16

**1、对HuggingFace原始权重进行W8A16量化**

请在先在 https://huggingface.co/bigscience/bloom/tree/main 下载 HuggingFace 官方权重到本地的 【HuggingFace权重路径】下，并准备一个用于存放量化权重的路径【W8A16权重路径】，并运行：

```shell
python handle_weights.py --handle-type quant --input-path 【HuggingFace权重路径】 --output-path 【W8A16权重路径】 --device 0 --world-size 8
```

**2、对W8A16量化权重进行TP并行切分**

完成对权重的量化后，还要准备一个路径用于存放切分后权重的路径【W8A16并行切分权重路径】，然后运行下面的命令：

```shell
python handle_weights.py --handle-type cut_quant --world-size 8 --input-path 【W8A16权重路径】 --output-path 【W8A16并行切分权重路径】
```

### 下游任务精度测试

#### FlashAttention模型

在 `ModelLink/mindie_ref/mindie_llm/atb_models/pytorch/examples/bloom/176b` 路径下：

```shell
export ATB_OPERATION_EXECUTE_ASYNC=1
torchrun --nproc_per_node 8 --master_port 11949 main.py --mode precision --model_path 【W8A16并行切分权重路径】 --data_dtype w8a16 --device 0 1 2 3 4 5 6 7 --hardware 910 > run.log
```

#### PagedAttention模型

在 `ModelLink/mindie_ref/mindie_llm/atb_models` 路径下：

```shell
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
bash tests/modeltest/run.sh pa_fp16 full_CEval 1 bloom_7b 【W8A16并行切分权重路径】 8
```

### 性能测试

#### FlashAttention模型

batch size 通过 `--batch batch_size` 来设置，默认运行会测试输入输出序列长度为 32、64、128、256、512、1024两两组合的测试结果：

```shell
export ATB_OPERATION_EXECUTE_ASYNC=1
torchrun --nproc_per_node 8 --master_port 11949 main.py --model_path 【W8A16并行切分权重路径】 --batch 1 --data_dtype w8a16 --device 0 1 2 3 4 5 6 7 --hardware 910 > run.log
```

可以通过 `--seqlen_in_pair` 选项来设置需要测试的 case，例如测试输入输出序列长度都是 512 的性能：

```shell
export ATB_OPERATION_EXECUTE_ASYNC=1
torchrun --nproc_per_node 8 --master_port 11949 main.py --model_path 【W8A16并行切分权重路径】 --batch 1 --seqlen_in_pair --seqlen_in_pair 512 --seqlen_out_pair 512 --data_dtype w8a16 --device 0 1 2 3 4 5 6 7 --hardware 910 > run.log
```
