# Mixtral

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [Mixtral](#mixtral)
- [目录](#目录)
- [8x7B](#8x7b)
  - [硬件要求](#硬件要求)
  - [准备工作](#准备工作)
  - [模型训练](#模型训练)
  - [模型性能](#模型性能)
    - [吞吐](#吞吐)

# 8x7B

## 硬件要求

训练的最低硬件配置:

| 硬件 |        配置        |
| :--: |:----------------:|
| NPU | 64 x Ascend NPUs |

推理的推荐硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

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
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 2b0edd2
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -r requirements.txt 
    ```

3. 下载 Mixtral-8x7B 的 [预训练权重和词表](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main)（*建议仅下载使用safetensors格式的权重*）

    ```shell
    #!/bin/bash
    cd ./model_from_hf/
    git lfs install
    git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
    mv Mixtral-8x7B-v0.1 Mixtral-8x7B
    cd ..
    ```

## 模型训练

1. 准备数据集

    下载 Mixtral-8x7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # 处理数据   
    mkdir ./dataset/Mixtral-8x7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
        --output-prefix ./dataset/Mixtral-8x7B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    配置 Mixtral-8x7B 预训练脚本：***examples/mixtral_mcore/pretrain_mixtral_8x7b_ptd.sh***

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    DATA_PATH="./dataset/Mixtral-8x7B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"
    CKPT_SAVE_DIR="./ckpt/Mixtral-8x7B/"

    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8
    MASTER_ADDR="your master node IP"
    MASTER_PORT=6000
    NNODES=8
    NODE_RANK="current node id"
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

    # 训练并行策略
    TP=8
    PP=2
    EP=1
    CP=4
    ```

    启动 Mixtral-8x7B 预训练脚本: ***examples/mixtral_mcore/pretrain_mixtral_8x7b_ptd.sh***

    ```shell
    bash examples/mixtral_mcore/pretrain_mixtral_8x7b_ptd.sh
    ```

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

## 模型性能

### 吞吐

Mixtral-8x7B 在双机16卡上(tp8 pp2) **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 |    模型     | tokens吞吐 (tokens/s/p) | 单步迭代时间 (s/step) |
| :--: |:---------:|:---------------------:|:---------------:|
| NPUs | Mixtral-8x7B |1568.3       |      20.9       |
| 参考 |   Mixtral-8x7B   |1766.6         |       18.5       |