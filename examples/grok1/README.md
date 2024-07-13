# Grok1 $\color{black}{\bf\tiny{【昇腾贡献模型】}}$

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [Grok1](#Grok1)
- [目录](#目录)
- [40B](#40B)
  - [硬件要求](#硬件要求)
  - [准备工作](#准备工作)
  - [模型训练](#模型训练)
  - [模型性能](#模型性能)
    - [吞吐](#吞吐)

# 40B

## 硬件要求

训练的最低硬件配置:

| 硬件 |        配置        |
| :--: |:----------------:|
| NPU | 16 x Ascend NPUs |

推理的推荐硬件配置:

| 硬件 |        配置        |
| :--: |:----------------:|
| NPU | 16 x Ascend NPUs |

## 准备工作

1. 克隆仓库到本地服务器

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../ModelLink/
    cd ..
    cd ModelLink
    mkdir logs
    mkdir model_from_hf/grok1-40B
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
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 2b0edd2
    pip install -r requirements.txt 
    pip install -e .
    cd ..

    # 安装其余依赖库
    pip install -r requirements.txt 
    ```

3. 下载 Grok1 40B 的 [模型文件和词表](https://huggingface.co/hpcai-tech/grok-1/tree/main)和[tokenizer.model](https://github.com/xai-org/grok-1/blob/main/tokenizer.model)

    ```shell
    #!/bin/bash
    cd ./model_from_hf/grok1-40B
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/config.json
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/configuration_grok1.py
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/modeling_grok1.py
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/modeling_grok1_outputs.py
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/special_tokens_map.json
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/tokenizer_config.json
    wget wget https://github.com/xai-org/grok-1/raw/main/tokenizer.model
    cd ..
    ```

## 模型训练

1. 准备数据集

    下载 Grok1 40B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # 处理数据   
    mkdir ./dataset/grok1-40B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/grok1-40B/ \
        --output-prefix ./dataset/grok1-40B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    配置 grok1-40B 预训练脚本：***examples/grok1/pretrain_grok1_40b_ptd.sh***

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    DATA_PATH="./dataset/grok1-40B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/grok1-40B/tokenizer.model"
    CKPT_SAVE_DIR="./ckpt/grok1-40B/"

    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8
    MASTER_ADDR="your master node IP"
    MASTER_PORT=6000
    NNODES=2
    NODE_RANK="current node id"
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

    # 训练并行策略
    TP=4
    PP=2
    EP=2
    ```

    启动 grok1-40B 预训练脚本: ***examples/grok1/pretrain_grok1_40b_ptd.sh***

    ```shell
    bash examples/grok1/pretrain_grok1_40b_ptd.sh
    ```

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。


## 模型性能

### 吞吐

grok1_40b 在两机16卡上(tp4 pp2 ep2) **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 |    模型     | tokens吞吐 (tokens/s/p) | 单步迭代时间 (s/step) |
| :--: |:---------:|:---------------------:|:---------------:|
| NPUs | grok1-40B |1646.633        |      39.8       |
| 参考 |   grok1-40B   |2057          |       32        |

