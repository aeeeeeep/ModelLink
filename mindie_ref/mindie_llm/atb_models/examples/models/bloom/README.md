# BLOOM

* [BLOOM](https://huggingface.co/bigscience/bloom) (BigScience Large Open-science Open-access Multilingual Language Model)
* 此代码仓中实现了一套基于 NPU 硬件的 BLOOM 推理模型。

## 特性矩阵

- 此矩阵罗列了各 BLOOM 模型支持的特性：

| 模型       | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 |
| ---------- | -------------------------- | --------------------------- | ---- | --------------- | --------------- | -------- | --------- |
| BLOOM-7B   | 支持 world size 8          | 支持 world size 2           | 是   | 是              | 是              | 是       | 否        |
| BLOOM-176B | 支持 world size 8          | 不支持                      | 是   | 是              | 是              | 否       | 是        |

## 推理使用说明

### 路径变量解释

| 变量名        | 含义                                                         |
| ------------- | ------------------------------------------------------------ |
| `working_dir` | 加速库及模型库下载后放置的目录                               |
| `llm_path`    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/ModelLink/`；若使用gitee下载的代码，则路径为`${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models` |
| `script_path` | 脚本所在路径。Baichuan系列模型的工作脚本所在路径为${llm_path}/examples/models/baichuan |
| `weight_path` | 模型权重路径                                                 |

## BLOOM-176B



## BLOOM-7B


