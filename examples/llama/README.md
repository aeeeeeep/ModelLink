# LLaMA
<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/ModelLink/blob/modellink/examples/llama/README_en.md">English</a></b>
</p>

- [LLaMA-7B/13B](#llama-7b13b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
    - [推理](#推理)
      - [脚本](#脚本)
    - [使用基线数据集进行评估](#使用基线数据集进行评估)
- [LLaMA-33B/65B](#llama-33b65b)
  - [训练](#训练)
    - [数据集](#数据集)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
    - [推理](#推理)
      - [脚本](#脚本)
    - [使用基线数据集进行评估](#使用基线数据集进行评估)
- [引用](#引用)

# LLaMA-7B/13B
## 训练

LLaMA-7B/13B 训练的硬件配置如下:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |


### 脚本

1. 拷贝仓库到个人服务器
   ```shell
   git clone https://gitee.com/ascend/ModelLink.git 
   cd ModelLink
   mkdir logs
   cd ..
   ```

2. 搭建环境

```bash
# python3.8
conda create -n test python=3.8
conda activate test
# 安装 torch 和 torch_npu
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.post5_XXXXXX-cp38-cp38m-linux_aarch64.whl
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 安装加速库
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
pip install -r requirements.txt
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt 
```

3. 下载 LLaMA-7B [权重和词表](https://huggingface.co/ruibin-wang/llama-7b-hf/tree/main) 或 LLaMA-13B [权重和词表](https://huggingface.co/ruibin-wang/llama-13b-hf/tree/main)
```shell
  mkdir model_from_hf
  cd ./model_from_hf
  # 需要安装 git-lfs
  git clone https://huggingface.co/ruibin-wang/llama-7b-hf
  cd ..
```
or 

```shell
  mkdir model_from_hf
  cd ./model_from_hf
  # 需要安装 git-lfs
  git clone https://huggingface.co/ruibin-wang/llama-13b-hf
  cd ..
```

将模型权重文件从 huggingface 格式转化为 megatron 格式

LLaMA-7B
```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
mkdir model_weights
python tools/checkpoint/util.py --model-type GPT \
                                --loader llama2_hf \
                                --saver megatron \
                                --target-tensor-parallel-size 1 \
                                --target-pipeline-parallel-size 8 \
                                --load-dir ./model_from_hf/llama-7b-hf \
                                --save-dir ./model_weights/llama-7b-tp1-pp8 \
                                --tokenizer-model ./model_from_hf/llama-7b-hf/tokenizer.model
cd ..
```

LLaMA-13B
```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
# 单机8卡
mkdir model_weights
python tools/checkpoint/util.py --model-type GPT \
                                --loader llama2_hf \
                                --saver megatron \
                                --target-tensor-parallel-size 1 \
                                --target-pipeline-parallel-size 8 \
                                --load-dir ./model_from_hf/llama-13b-hf \
                                --save-dir ./model_weights/llama-13b-tp1-pp8 \
                                --tokenizer-model ./model_from_hf/llama-13b-hf/tokenizer.model
```

权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数`target-tensor-parallel-size`和`target-pipeline-parallel-size`。

4. 预训练

4.1 准备预训练数据集

下载 LLaMA-7B/13B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

处理数据集

LLaMA-7B
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-7b-hf \
    --output-prefix ./dataset/llama \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF  
```

LLaMA-13B
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-13b-hf \
    --output-prefix ./dataset/llama \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF  
```

4.2 配置 LLaMA-7B/13B 预训练脚本

LLaMA-7B
```shell
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集路径，权重路径，词表路径等
TOKENIZER_MODEL=./model_from_hf/llama-7b-hf/tokenizer.model
DATA_PATH=./dataset/llama_text_document  #数据集 路径
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

LLaMA-13B
```shell
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集路径，权重路径，词表路径等
TOKENIZER_MODEL=./model_from_hf/llama-13b-hf/tokenizer.model 
DATA_PATH=./dataset/llama_text_document  #数据集 路径
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

4.3 启动 LLaMA-7B/13B 预训练脚本

LLaMA-7B
```shell
bash examples/llama/pretrain_llama_7b_ptd.sh
```

LLaMA-13B
```shell
# 单机8卡
bash examples/llama/pretrain_llama_13b_ptd.sh 
```

5. 微调

5.1 准备微调数据集

下载 LLaMA-7B/13B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
mkdir finetune_dataset
cd ./finetune_dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

处理指令数据集  

LLaMA-7B
```shell                          
python ./tools/preprocess_data.py \
  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/llama-7b-hf \
  --output-prefix ./finetune_dataset/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralInstructionHandler \
  --append-eod
```

LLaMA-13B
```shell                          
python ./tools/preprocess_data.py \
  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/llama-13b-hf \ 
  --output-prefix ./finetune_dataset/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralInstructionHandler \
  --append-eod
```

5.2 配置 LLaMA-7B/13B 微调脚本

LLaMA-7B
```shell
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集路径，权重路径，词表路径等
TOKENIZER_PATH=./model_from_hf/llama-7b-hf/  #tokenizer 路径
DATA_PATH=./finetune_dataset/alpaca  #数据集 路径
LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

LLaMA-13B
```shell
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集路径，权重路径，词表路径等
TOKENIZER_PATH=./model_from_hf/llama-13b-hf/  #tokenizer 路径
DATA_PATH=./finetune_dataset/alpaca  #数据集 路径
LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

增加微调参数--finetune，使微调从第一步开始。

5.3 启动 LLaMA-7B/13B 微调脚本

LLaMA-7B
```shell
bash examples/llama/tune_llama_7b_ptd.sh
```

LLaMA-13B
```shell
# 单机8卡
bash examples/llama/tune_llama_13b_ptd.sh 
```

### 性能

#### 吞吐

LLaMA-7B/13B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备   | 硬件        | 模型        | 迭代数  | 样本吞吐 (samples/p/s) | token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 浮点计算数 (TFLOPs/s) |
|------|-----------|-----------|------|--------------------|----------------------|-----------------|------------------|
| NPUs | 910 1*8p  | LLaMA-7B  | 2048 | 1.75               | 3600                 | 18.2            | 159.9            |
| 参考 | -         | LLaMA-7B  | 2048 | 1.85               | 3804                 | 18.5            | 161.5            |
| NPUs | 910 1*8p  | LLaMA-13B | 2048 | 0.92               | 1895                 | 17.2            | 200.57           |
| 参考 | -         | LLaMA-13B | 2048 | 0.96               | 2012                 | 16.65           | 213.29           |




## 推理

我们支持使用 LLaMA-7B 和 LLaMA-13B 进行文本生成的推理。
推理与预训练不同，比如我们需要加载预训练权重和输出样本的长度：

配置LLaMA-7B推理脚本`tasks/inference/generate_llama_7b_ptd.sh`和LLaMA-13B推理脚本`tasks/inference/generate_llama_13b_ptd.sh`。

```shell
# 修改模型权重路径和分词器路径
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
```

LLaMA-7B:
```shell
bash ./tasks/inference/generate_llama_7b_ptd.sh
```

LLaMA-13B:
```shell
bash ./tasks/inference/generate_llama_13b_ptd.sh
```

配置LLaMA-7B lora推理脚本`tasks/inference/generate_llama_7b_lora_ptd.sh`和LLaMA-13B lora推理脚本`tasks/inference/generate_llama_13b_lora_ptd.sh`。
```bash
# 修改lora权重路径
CHECKPOINT_LORA="your lora model directory path"
```
LLaMA-7B:
```shell
bash ./tasks/inference/generate_llama_7b_lora_ptd.sh
```

LLaMA-13B:
```shell
bash ./tasks/inference/generate_llama_13b_lora_ptd.sh
```

部分推理样本如下：

LLaMA-7B:

![llama-7B_generate.png](../../sources/images/llama/llama-7B_generate.png)

LLaMA-13B:

![llama-13B_generate.png](../../sources/images/llama/llama-13B_generate.png)


## 使用基线数据集进行评估

我们使用 Boolq benchmark 来评估我们的模型。Benchmark下载[此处](https://huggingface.co/datasets/boolq)。


配置LLaMA-7B评估脚本 `tasks/evaluation/evaluate_llama_7B_ptd.sh` 和 LLaMA-13B评估脚本 `tasks/evaluation/evaluate_llama_13B_ptd.sh`：

修改权重路径, 词表路径和数据集任务路径：
```shell
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
DATA_PATH="./boolq/data/test/"
TASK="boolq"
```

修改最大生成词参数：
```shell
--max-new-tokens 1 
```

开始评估：
```shell
bash tasks/evaluation/evaluate_llama_7B_ptd.sh
bash tasks/evaluation/evaluate_llama_13B_ptd.sh
```

LLaMA-7B/13B在**Ascend NPU**中的评测表现：

| 任务                                                  | 模型        | 昇腾值  | 社区值  |
|-----------------------------------------------------|-----------|------|------|
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-7B  | 74.6 | [75.4](https://hub.opencompass.org.cn/dataset-detail/BoolQ) | 
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-13B | 79.6 | [78.7](https://hub.opencompass.org.cn/dataset-detail/BoolQ) |

# LLaMA-33B/65B

LLaMA 模型源于: [LLaMA: OPen and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971v1.pdf)

>Touvron, Hugo, et al. "LLaMA: OPen and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971 (2023).

## 训练

LLaMA-33B/65B 训练的硬件配置:

|  硬件 |        配置        |
|:---:|:----------------:|
| NPU | 32 x Ascend NPUs |

### 数据集
模型使用 alpaca 数据集训练

### 脚本
1. 拷贝仓库到个人服务器
   ```shell
   git clone https://gitee.com/ascend/ModelLink.git 
   cd ModelLink
   mkdir logs
   cd ..
   ```
   
2. 搭建环境
```shell
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu
# ARM
wget https://download.pytorch.org/whl/torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0.post4_XXXXXX-cp38-cp38m-manylinux2014_aarch64.whl
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装加速库
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt 
```

3. 下载权重和词表

llama-33B 权重
```shell
mkdir tokenizer
cd ./tokenizer

# 需要安装 git-lfs
git lfs install
git clone https://huggingface.co/pinkmanlove/llama-33b-hf
cd ..
# 将 tokenizer_config.json 中的 "LLaMATokenizer" 修改为 "LLaMTokenizer" （这是hf的一个bug）
```

llama-65B 权重
```shell
mkdir model_from_hf
cd ./model_from_hf

# 需要安装 git-lfs
git lfs install
git clone https://huggingface.co/pinkmanlove/llama-65b-hf
cd ..
# 将 tokenizer_config.json 中的 "LLaMATokenizer" 修改为 "LLaMTokenizer" （这是hf的一个bug）
```

4. 预训练权重从 huggingface 格式转换为 megatron 格式

llama-33B
```shell
mkdir model_weights

SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
      --input-model-dir ./tokenizer \
      --output-model-dir ./model_weights \
      --tensor-model-parallel-size 4 \
      --pipeline-model-parallel-size 4 \
      --merge-mlp \
      --type 30B
```

llama-65B
```shell
mkdir model_weights
python tools/checkpoint/util.py --model-type GPT \
                                --loader llama2_hf \
                                --saver megatron \
                                --target-tensor-parallel-size 8 \
                                --target-pipeline-parallel-size 4 \
                                --load-dir ./model_from_hf/llama-65b-hf \
                                --save-dir ./model_weights/llama-65b-tp8-pp4 \
                                --tokenizer-model ./model_from_hf/llama-65b-hf/tokenizer.model
```

权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数`target-tensor-parallel-size`和`target-pipeline-parallel-size`。

5. 预训练

5.1 准备预训练数据集

```shell
# 下载 alpaca 数据集
wget http://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
```

处理数据集

LLaMA-33B
```shell
mkdir dataset
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-33b-hf \
    --output-prefix ./dataset/llama \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF 
```

LLaMA-65B
```shell
mkdir dataset
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-65b-hf \
    --output-prefix ./dataset/llama \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF 
```

5.2 配置 LLaMA-33B/65B 预训练脚本

LLaMA-33B
```bash
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 配置词表和数据路径等
TOKENIZER_MODEL=./tokenizer/llama-33b-hf/tokenizer.model
DATA_PATH=./dataset/llama_text_document 
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

LLaMA-65B
```bash
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 配置词表和数据路径等
TOKENIZER_MODEL=./model_from_hf/llama-65b-hf/tokenizer.model
DATA_PATH=./dataset/llama_text_document 
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

5.3 启动预训练脚本:

启动 llama-33B 预训练脚本 : ./examples/llama/pretrain_llama_33B_ptd_32p.sh
```bash
bash examples/llama/pretrain_llama_33B_ptd_32p.sh
```

启动 llama-65B 预训练脚本 : ./examples/llama/pretrain_llama_65b_ptd.sh
```bash
# 四机32卡
bash examples/llama/pretrain_llama_65b_ptd.sh
```

为多节点配置 llama-33B/65B 预训练脚本 (在集群的每个节点上启动脚本):

```shell
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=4
NODE_RANK=0
```

训练log如下:

```Shell
 iteration  11/50000 | consumed samples: 5632 | consumed tokens:  11534336 | elapsed time per iteration (ms):  52728.1 | learning rate:    1.499E-05 | gloabl batch size:  512 | lm loss:  1.376514E+01 | loss scale:  65536.0 | grad norm:    459.628 | actual seqlen:  2048 | number of skipped
iterations: 0 | number of nan iterations:   0 | samples per second: 9.710 | TFLOPs: 167.52 |
time (ms)
```

6. 微调

6.1 准备微调数据集

下载数据集

```shell
mkdir finetune_dataset
cd ./finetune_dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

处理指令数据集

LLaMA-33B
```shell                           
python ./tools/preprocess_data.py \
  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./tokenizer/llama-33b-hf \ 
  --output-prefix ./finetune_dataset/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralInstructionHandler \
  --append-eod
```

LLaMA-65B
```shell                           
python ./tools/preprocess_data.py \
  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/llama-65b-hf \
  --output-prefix ./finetune_dataset/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralInstructionHandler \
  --append-eod
```

6.2 配置 LLaMA-33B/65B 微调脚本

LLaMA-33B
```shell
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集路径，权重路径，词表路径等
TOKENIZER_PATH=./tokenizer/llama-33b-hf/  #tokenizer 路径
DATA_PATH=./finetune_dataset/alpaca  #数据集 路径
LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

LLaMA-65B
```shell
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集路径，权重路径，词表路径等
TOKENIZER_PATH=./model_from_hf/llama-65b-hf/  #tokenizer 路径
DATA_PATH=./finetune_dataset/alpaca  #数据集 路径
LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

增加微调参数--finetune，使微调从第一步开始。

6.3 启动 LLaMA-33B/65B 微调脚本

LLaMA-33B
```shell
bash examples/llama/tune_llama_33b_ptd.sh
```

LLaMA-65B
```shell
# 双机16卡
bash examples/llama/tune_llama_65b_ptd.sh 
```

### 性能

#### 吞吐

LLaMA-33B/65B在 **昇腾芯片** 和 **参考芯片** 上的性能对比:

|  设备  |    模型     | tokens吞吐 (tokens/s/p) |
|:----:|:---------:|:---------------------:|
|  参考  | llama-33B |          776          |
| NPUs | llama-33B |          621          |
|  参考  | llama-65B |          426          |
| NPUs | llama-65B |          348          |



## 推理

我们支持使用 LLaMA-33B 和 LLaMA-65B 进行文本生成的推理。
推理与预训练不同，比如我们需要加载预训练权重和输出样本的长度：

配置LLaMA-33B推理脚本`tasks/inference/generate_llama_33B_ptd.sh`。

配置LLaMA-65B推理脚本`tasks/inference/generate_llama_65B_ptd.sh`。

```shell
# 修改模型权重路径和分词器路径
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
```

LLaMA-33B:
```shell
bash ./tasks/inference/generate_llama_33B_ptd.sh
```
LLaMA-65B:
```shell
bash ./tasks/inference/generate_llama_65B_ptd.sh
```

配置LLaMA-33B lora推理脚本`tasks/inference/generate_llama_33B_lora_ptd.sh`。

配置LLaMA-65B lora推理脚本`tasks/inference/generate_llama_65B_lora_ptd.sh`。

```bash
# 修改lora权重路径
CHECKPOINT_LORA="your lora model directory path"
```
LLaMA-33B:
```shell
bash ./tasks/inference/generate_llama_33b_lora_ptd.sh
```

LLaMA-65B:
```shell
bash ./tasks/inference/generate_llama_65b_lora_ptd.sh
```

部分推理样本如下：

LLaMA-33B:

![llama-13B_generate.png](../../sources/images/llama/llama33B_generate.png)

LLaMA-65B:

![llama-65B_generate.png](../../sources/images/llama/llama-65B_generate.png)

## 使用基线数据集进行评估

我们使用 Boolq benchmark 来评估我们的模型。Benchmark下载[此处](https://huggingface.co/datasets/boolq)。

配置LLaMA-33B评估脚本：tasks/evaluation/evaluate_llama_33B_ptd.sh

配置LLaMA-65B评估脚本：tasks/evaluation/evaluate_llama_65B_ptd.sh

修改权重路径, 词表路径和数据集任务路径：
```shell
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
DATA_PATH="./boolq/data/test/"
TASK="boolq"
```

修改最大生成词参数：
```shell
--max-new-tokens 1 
```
开始评估

```shell
# llama-33B评估
bash tasks/evaluation/evaluate_llama_33B_ptd.sh
# llama-65B评估
bash tasks/evaluation/evaluate_llama_65B_ptd.sh
```

LLaMA-33B和LLaMA-65B在**Ascend NPU**中的评测表现：

| 任务                                             | 模型        | 昇腾值  | 社区值                                                                 |
|------------------------------------------------|-----------|------|---------------------------------------------------------------------|
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-33B | 83.2 | [83.1](https://paperswithcode.com/sota/question-answering-on-boolq) |
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-65B | 85.7 | [86.6](https://paperswithcode.com/sota/question-answering-on-boolq) |

## 引用


```shell
@article{Touvron2023llama,
  title={LLaMA: OPen and Efficient Foundation Language Models},
  author={Hugo Touvron*, Thibaut Lavril*, Gautier Izacard*, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal,
  Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave*, Guillaume Lample*},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}}
```