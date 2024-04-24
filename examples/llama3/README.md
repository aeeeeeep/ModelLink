# LLaMA3

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [LLaMA3](#llama3)
- [目录](#目录)
- [LLAMA3-8B](#llama3-8b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
  - [推理-8B](#推理-8b)
  - [评估-8B](#评估-8b)
- [LLAMA3-70B](#llama3-70b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
  - [推理-70B](#推理-70b)
  - [评估-70B](#评估-70b)

# LLAMA3-8B

## 训练

LLAMA3-8B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器

   ```shell
   git clone https://gitee.com/ascend/ModelLink.git 
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout -f bcce6f
   cp -r megatron ../ModelLink/
   cd ..
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
   pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
   pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
   pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh 

   # 安装加速库
   git clone https://gitee.com/ascend/AscendSpeed.git
   cd AscendSpeed
   git checkout 224ae35e8fc96778f957029d1371ddb623452a50
   pip install -r requirements.txt 
   pip3 install -e .
   cd ..

   # 安装其余依赖库
   pip install -r requirements.txt 
   ```
3. 下载 LLAMA3-8B 的 [预训练权重和词表](https://huggingface.co/unsloth/llama-3-8B/tree/main)

   ```shell
     #!/bin/bash
     mkdir ./model_from_hf/llama-3-8b-hf/
     cd ./model_from_hf/llama-3-8b-hf/
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/config.json
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/generation_config.json
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/model-00001-of-00004.safetensors
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/model-00002-of-00004.safetensors
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/model-00003-of-00004.safetensors
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/model-00004-of-00004.safetensors
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/model.safetensors.index.json
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/special_tokens_map.json
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/tokenizer.json
     wget https://huggingface.co/unsloth/llama-3-8B/blob/main/tokenizer_config.json
     cd ../../
   ```
4. 权重转换

   4.1 将权重从 huggingface 格式转化为 magatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/llama-3-8b-hf/ \
        --save-dir ./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/llama-3-8b-hf/tokenizer.json
   ```

   4.2 任意并行切分策略的 Megatron 权重 格式转化为 HuggingFace权重
   ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-3-8b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-3-8b-hf/mg2hg/
```

   权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。

5. 预训练

   5.1 准备数据集

   下载 LLaMA3-8B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
     # 下载数据
     cd ./dataset
     wget https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     # 处理数据   
     mkdir ./dataset/llama-3-8b-hf/
     python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/llama-3-8b-hf/ \
       --output-prefix ./dataset/llama-3-8b-hf/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF
   ```

   5.2 预训练

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/llama-3-8b-hf/"
    TOKENIZER_MODEL="./model_from_hf/llama-3-8b-hf/"  #词表路径
    DATA_PATH="./dataset/llama-3-8b-hf/alpaca_text_document"  #数据集路径
   ```

   多机运行增加参数--overlap-grad-reduce

   启动 LLaMA3-8B 预训练脚本: examples/llama3/pretrain_llama3_8b_ptd.sh

   ```shell
    bash examples/llama3/pretrain_llama3_8b_ptd.sh
   ```

   **注意**：如果使用多机训练，需要设置多机数据共享，非主节点通过数据共享读取主节点数据。或者，直接将主节点生成的数据复制到非主节点。

### 性能

#### 吞吐

LLaMA3-8B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 |   模型   | 迭代数 | tokens吞吐 (tokens/s/p) |
| :--: | :-------: | :----: | :---------------------: |
| NPUs | LLaMA3-8B |  1000  |        2275          |
| 参考 | LLaMA3-8B |  1000  |        2570          |
## 推理-8B

配置llama3-8B 推理脚本: examples/llama3/generate_llama3_8b_ptd.sh

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"
TOKENIZER_MODEL="./model_from_hf/llama-3-8b-hf/tokenizer.json"
```

启动llama3-8B 推理脚本

```bash
bash examples/llama3/generate_llama3_8b_ptd.sh
```


## 评估-8B

使用 MMLU数据集评估模型. 数据集下载路径 [这里](https://huggingface.co/datasets/cais/mmlu).
配置llama3-8B 评估脚本: examples/llama3/evaluate_llama3_8B_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"  #词表路径
CHECKPOINT="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/llama3/evaluate_llama3_8B_ptd.sh
```

评估结果如下

| 数据集 | 总学科数 | 总问题数 | 参考准确率 | NPU准确率 |
| :----: | :------: | :------: | :--------: | :-------: |
|  MMLU  |    57    |  14042  |   0.666   |  0.653  |

# LLAMA3-70B

## 训练

LLAMA3-70B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 64 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器

   ```shell
   git clone https://gitee.com/ascend/ModelLink.git 
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout -f bcce6f
   cp -r megatron ../ModelLink/
   cd ..
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
   pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
   pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
   pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh 

   # 安装加速库
   git clone https://gitee.com/ascend/AscendSpeed.git
   cd AscendSpeed
   git checkout 224ae35e8fc96778f957029d1371ddb623452a50
   pip install -r requirements.txt 
   pip3 install -e .
   cd ..

   # 安装其余依赖库
   pip install -r requirements.txt 
   ```
3. 下载 LLAMA3-70B 的 [预训练权重和词表](https://huggingface.co/v2ray/Llama-3-70B/tree/main)

   ```shell
     #!/bin/bash
     mkdir ./model_from_hf/llama-3-70b-hf/
     cd ./model_from_hf/llama-3-70b-hf/
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/config.json
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/generation_config.json
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00001-of-00030.safetensors
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00002-of-00030.safetensors
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00003-of-00030.safetensors
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00004-of-00030.safetensors
     ...
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00030-of-00030.safetensors
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model.safetensors.index.json
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/special_tokens_map.json
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/tokenizer.json
     wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/tokenizer_config.json
     cd ../../
   ```
4. 权重转换

   4.1 将权重从 huggingface 格式转化为 magatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 8 \
        --load-dir ./model_from_hf/llama-3-70b-hf/ \
        --save-dir ./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/ \
        --tokenizer-model ./model_from_hf/llama-3-70b-hf/tokenizer.json
   ```

   4.2 任意并行切分策略的 Megatron 权重 格式转化为 HuggingFace权重
   ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-3-70b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-3-70b-hf/mg2hg/
```

   权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。

5. 预训练

   5.1 准备数据集

   下载 LLaMA3-70B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
     # 下载数据
     cd ./dataset
     wget https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     # 处理数据   
     mkdir ./dataset/llama-3-70b-hf/
     python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/llama-3-70b-hf/ \
       --output-prefix ./dataset/llama-3-70b-hf/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF
   ```

   5.2 预训练

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/llama-3-70b-hf/"
    TOKENIZER_MODEL="./model_from_hf/llama-3-70b-hf/"  #词表路径
    DATA_PATH="./dataset/llama-3-70b-hf/alpaca_text_document"  #数据集路径
   ```

   多机运行增加参数--overlap-grad-reduce

   启动 LLaMA3-70B 预训练脚本: examples/llama3/pretrain_llama3_70b_ptd.sh

   ```shell
    bash examples/llama3/pretrain_llama3_70b_ptd.sh
   ```

   **注意**：如果使用多机训练，需要设置多机数据共享，非主节点通过数据共享读取主节点数据。或者，直接将主节点生成的数据复制到非主节点。

### 性能

#### 吞吐

LLaMA3-70B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 |   模型   | 迭代数 | tokens吞吐 (tokens/s/p) |
| :--: | :-------: | :----: | :---------------------: |
| NPUs | LLaMA3-70B |  1000  |        283          |
| 参考 | LLaMA3-70B |  -  |        -          |
## 推理-70B

配置llama3-70B 推理脚本: examples/llama3/generate_llama3_70b_ptd.sh

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/llama-3-70b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"
TOKENIZER_MODEL="./model_from_hf/llama-3-70b-hf/tokenizer.json"
```

启动llama3-70B 推理脚本

```bash
bash examples/llama3/generate_llama3_70b_ptd.sh
```

## 评估-70B

使用 MMLU数据集评估模型. 数据集下载路径 [这里](https://huggingface.co/datasets/cais/mmlu).
配置llama3-70B 评估脚本: examples/llama3/evaluate_llama3_70B_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"  #词表路径
CHECKPOINT="./model_weights/llama-3-70b-hf-v0.1-tp8-pp1"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/llama3/evaluate_llama3_70B_ptd.sh
```

评估结果如下

| 数据集 | 总学科数 | 总问题数 | 参考准确率 | NPU准确率 |
| :----: | :------: | :------: | :--------: | :-------: |
|  MMLU  |    57    |  14042  |   0.795   |  0.783  |