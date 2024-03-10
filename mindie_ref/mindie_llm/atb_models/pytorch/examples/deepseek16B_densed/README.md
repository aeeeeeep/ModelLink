# Deepseek16B 推理指导

### 快速运行
#### 1. 设置环境变量
```shell
# source cann环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source 加速库环境变量
source /usr/local/Ascend/atb/set_env.sh
# source 模型仓tar包解压出来后的环境变量
source set_env.sh
```
#### 2. 绑NUMA
```shell
# 安装绑numa工具：
yum install numactl
# 查询Bus-Id
npu-smi info
# 查询每张卡的NUMA node
lspci -vs Bus-Id
# 修改pytorch/examples/deepseek16B_densed/cut_model_and_run_mixtral.sh文件中的map映射关系
map["device id"]="numa node id"
```
#### 3. 标准性能测试
使用的是 HuggingFace 权重
```shell
# 进入范例脚本
cd pytorch/examples/deepseek16B_densed/
# 执行推理前需修改权重路径，将pytorch/examples/deepseek16B_densed/cut_model_and_run_mixtral.sh文件中的weight_dir="/home/data/acltransformer_testdata/weights/Mixtral-8x7B-v0.1-part-model-8"修改为个人权重路径
# 执行以下命令，可以快速进行性能测试，测试结果保存在与cut_model_and_run_mixtral.sh同级目录下的的mixtral_performance文件夹下
bash cut_model_and_run_mixtral.sh 
```
#### 4. 自定义规格性能测试
```shell
# 1、修改pytorch/examples/deepseek16B_densed/config.json中的 max_position_embeddings，推理过程中输入输出的总序列长度不可以超过该长度
# 2、修改pytorch/examples/deepseek16B_densed/run_mixtral.py中的batch_list,seq_in_list,seq_out_list
# 注1：batch_list不可以放太多，会导致内存碎片过多，一般使用场景是固定batch数量
# 注2：warm up的输入长度要设置为最大规格，后面推理的输入长度不应该超过改长度
batch_list = [28, 24, 16, 8, 4, 2, 1]
seq_in_list = [1024] 
seq_out_list = [250] 
```
#### 5. 对话测试
```shell
# 1、修改pytorch/examples/deepseek16B_densed/run_mixtral.py中的multi_prompt为用户自己的输入
```

## 精度测试
### 1. 前置条件
##### 下载[mmlu dataset](https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_chat_mmlu.py)并解压到pytorch/examples/deepseek16B_densed路径下
##### 下载[TruthfulQA dataset](https://github.com/sylinrl/TruthfulQA/tree/main)并解压到pytorch/examples/deepseek16B_densed路径下
### 2. 修改模型基础配置文件位置信息
```shell
# 注：模型实际读取的配置文件信息位于模型权重文件夹中，如上地址文件在每次模型执行时会被拷贝并替换权重文件夹中的原生config.json文件。使用此种方式旨在于方便用户随时对所有权重文件夹下的config.json进行同步修改
{
  "architectures": [
    "MixtralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  "model_type": "mixtral",
  "num_attention_heads": 32,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "num_local_experts": 8,
  "output_router_logits": false,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "router_aux_loss_coef": 0.02,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.36.2",
  "use_cache": true,
  "vocab_size": 32000,
  "world_size": 8
}
```
### 3. 使用加速库，评估下游任务精度表现（mmlu为例）
```shell
# 修改cut_model_and_run_mixtral.sh的99行改为如下：
# 将
numactl --cpunodebind=$bind --membind $bind python3 run_mixtral.py --load_path $weight_dir &
# 替换为
numactl --cpunodebind=$bind --membind $bind python3 eval/evaluate_mmlu.py -d data/mmlu/data/ --load_path $weight_dir &
# 然后执行
bash cut_model_and_run_mixtral.sh
```

### 4. 使用加速库，评估下游任务精度表现（TruthfulQA(mc1, mc2)为例）
```shell
# 完成前置条件：下载[TruthfulQA dataset](https://github.com/sylinrl/TruthfulQA/tree/main)并解压到pytorch/examples/deepseek16B_densed路径下(如果文件夹名为TruthfulQA-main则改为TruthfulQA)
# 将pytorch/examples/deepseek16B_densed/truthfulqa/main.py移动到pytorch/examples/deepseek16B_densed/TruthfulQA里
# 将pytorch/examples/deepseek16B_densed/truthfulqa文件夹替换掉pytorch/examples/deepseek16B_densed/TruthfulQA/truthfulqa
# 修改cut_model_and_run_mixtral.sh的99行改为如下：
# 将
numactl --cpunodebind=$bind --membind $bind python3 run_mixtral.py --load_path $output_dir &
# 替换为
numactl --cpunodebind=$bind --membind $bind python3 ./TruthfulQA/main.py --models mixtral --metrics mc --input_path ./TruthfulQA/TruthfulQA.csv --output_path ./TruthfulQA/TruthfulQA_answers.csv --load_path $output_dir &
# 然后执行
bash cut_model_and_run_mixtral.sh
# 查看结果可见同级目录下summary.csv
```

