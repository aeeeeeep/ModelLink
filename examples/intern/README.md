# Intern-LM
<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
    </p>
</p>

# 目录

- [Internlm-7B](#internlm-7b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
    - [推理](#推理)
    - [评估](#评估)
- [Internlm-65B](#internlm-65b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)

# InternLM-7B

## 训练

InternLM-7B 训练的硬件配置如下:

|  硬件 |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |


### 脚本

1. 拷贝仓库到本地服务器
```shell
git clone https://gitee.com/ascend/ModelLink.git
cd ModelLink
mkdir logs
mkdir model_from_hf
mkdir dataset
mkdir ckpt
```

2. 搭建环境

```bash
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu 
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_XXX.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

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

3. 下载 Internlm-7B [词表文件](https://huggingface.co/internlm/internlm-7b/tree/main)

```shell
cd ./model_from_hf/
# 需要安装 git-lfs: git lfs install
git clone https://huggingface.co/internlm/internlm-7b
cd ..
```

4. 下载 Internlm-7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

```shell
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/internlm-7b/ \
    --output-prefix ./dataset/internlm-7b_alpaca \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF  \
    --handler-name AlpacaPretrainHandler  \
    --tokenizer-not-use-fast \
    --append-eod
```

5. 权重格式转换

将模型权重文件从 HuggingFace权重 格式转化为 Megatron 权重
***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

```shell
python tools/checkpoint/util.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/internlm-7b/ \
    --save-dir ./model_weights/internlm-7b-v0.1-tp8-pp1/ \
    --tokenizer-model ./model_from_hf/internlm-7b/tokenizer.model \
    --add-qkv-bias \
    --add-dense-bias
```

任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/internlm-7b-v0.1-tp8-pp1/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --add-dense-bias \
    --save-dir ./model_from_hf/internlm-7b/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/internlm-7b/mg2hg
```



6. 配置 Internlm-7B 预训练脚本

```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集，词表，权重等路径
CKPT_SAVE_DIR="./ckpt/"
CKPT_LOAD_DIR="./model_weights/internlm-7b-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/internlm-7b/tokenizer.model" #词表路径
DATA_PATH="./dataset/internlm-7b_alpaca_text_document" #数据集路径
```

7. 启动 Internlm-7B 预训练脚本

```shell
bash examples/intern/pretrain_internlm_7b_ptd.sh 
```


### 性能

#### 吞吐

Internlm-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 | 模型          | 总迭代数 | 样本吞吐 (samples/s) | token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 
|----|-------------|------|--------------------|----------------------|-----------------|
| NPUs | Internlm-7B | 1000 | 10.85            | 2776                 | 5.90       |
| 参考 | Internlm-7B | 1000 | 11.14              | 2854                 |  5.74             | 


#### 推理
推理脚本</a>：
tasks/inference/generate_lnternlm_7b_ptd.sh
```
bash ./tasks/inference/generate_lnternlm_7b_ptd.sh
```
推理举例：
![Internlm-7b-inference](../../sources/images/intern/intern_7B_inference.png)

#### 评估
使用MMLU数据集评估模型。数据集[下载](https://huggingface.co/datasets/cais/mmlu)

评估脚本</a>:
tasks/evaluation/evaluate_internlm_7B_ptd.sh 
```
bash  tasks/evaluation/evaluate_internlm_7B_ptd.sh
```
InternLM-7B在**Ascend NPU**中的评测表现：

| 任务                                                  | 模型        | 昇腾值  | 社区值  |
|-----------------------------------------------------|-----------|------|------|
| MMLU | Internlm-7B  | 48.7 | [51.0](https://huggingface.co/internlm/internlm-7b) | 

# InternLM-65B

## 训练

InternLM-65B 训练的硬件配置如下:

|  硬件 |       配置        |
|:---:|:---------------:|
| NPU | 32 x Ascend NPUs |


### 脚本

1. 拷贝仓库到本地服务器
```shell
git clone https://gitee.com/ascend/ModelLink.git
cd ModelLink
mkdir logs
mkdir model_from_hf
mkdir dataset
mkdir ckpt
```

2. 搭建环境

```bash
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu 
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_XXX.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

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

3. 下载 [词表文件](https://huggingface.co/internlm/internlm-7b/tree/main)

```shell
#!/bin/bash
mkdir ./model_from_hf/internlm-7b/
cd ./model_from_hf/internlm-7b/
wget https://huggingface.co/internlm/internlm-7b/resolve/main/config.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/generation_config.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/special_tokens_map.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenization_internlm.py
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer.model
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer_config.json
cd ../../
```

4. 下载 Internlm-65B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

```shell
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/internlm-7b/ \
    --output-prefix ./dataset/internlm-7b_alpaca \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF  \
    --handler-name AlpacaPretrainHandler  \
    --tokenizer-not-use-fast \
    --append-eod
```

5. 配置 Internlm-65B 预训练脚本

```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集，词表，权重等路径
CKPT_SAVE_DIR="./ckpt/"
TOKENIZER_PATH="./model_from_hf/internlm-7b/" #词表路径
DATA_PATH="./dataset/internlm-7b_alpaca_text_document" #数据集路径
```

6. 启动 Internlm-65B 预训练脚本

```shell
bash examples/intern/pretrain_internlm_65b_ptd.sh 
```


### 性能

#### 吞吐

Internlm-65B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 | 模型          | 总迭代数 | 样本吞吐 (samples/p/s) | token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 
|----|-------------|------|--------------------|----------------------|-----------------|
| NPUs | Internlm-65B |  |      5.33      |            341    |     48     | 
| Reference | Internlm-65B | - | -              | 414                 | -            | 
