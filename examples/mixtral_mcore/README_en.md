# Mixtral

<p align="left">
        <b><a href="README.md">简体中文</a> </b> |
        <b>English</b> 
</p>

# Table of Contents

- [Mixtral](#mixtral)
- [Table of Contents](#table-of-contents)
- [8x7B](#8x7b)
  - [Hardware-Requirements](#hardware-requirements)
  - [Preparation](#preparation)
  - [Model-Training](#model-training)
  - [Model-Performance](#model-performance)
    - [Throughput](#throughput)

# 8x7B

## Hardware-Requirements

Minimum hardware requirements for training:

| Hardware |  Configuration   |
| :------: |:----------------:|
|   NPU   | 64 x Ascend NPUs |

Recommended hardware configuration for inference:

| Hardware |  Configuration  |
| :------: | :-------------: |
|   NPU   | 8 x Ascend NPUs |

## Preparation

1. Clone the code repository to the local server

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

2. Set up the environment

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # Install torch and torch_npu
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # modify the path according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # install MindSpeed
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 2b0edd2
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```

3. Download the pre-trained weights and vocabulary for Mixtral-8x7B from [here](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main). (It is recommended to only download weights in safetensors format)

    ```shell
    #!/bin/bash
    cd ./model_from_hf/
    git lfs install
    git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
    mv Mixtral-8x7B-v0.1 Mixtral-8x7B
    cd ..
    ```

## Model-Training

1. Prepare dataset

    Download the datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet), save to ModelLink/dataset/ directory.

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # process datasets
    mkdir ./dataset/Mixtral-8x7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
        --output-prefix ./dataset/Mixtral-8x7B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    Configure Mixtral-8x7B pre-training script: ***examples/mixtral_mcore/pretrain_mixtral_8x7b_ptd.sh***

    ```shell
    # Set the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # Configure according to the actual vocabulary, dataset, and model parameter save path
    DATA_PATH="./dataset/Mixtral-8x7B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"
    CKPT_SAVE_DIR="./ckpt/Mixtral-8x7B/"

    # Configure distributed parameters according to the actual distributed cluster
    GPUS_PER_NODE=8
    MASTER_ADDR="your master node IP"
    MASTER_PORT=6000
    NNODES=8
    NODE_RANK="current node id"
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

    # Training parallel strategy
    TP=8
    PP=2
    EP=1
    CP=4
    ```

    Start Mixtral-8x7B pre-training script: ***examples/pretrain_mixtral_8x7b_ptd.sh***

    ```shell
    bash examples/mixtral_mcore/pretrain_mixtral_8x7b_ptd.sh
    ```

    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

## Model-Performance

### Throughput

Comparison of Mixtral-8x7B performance on 2 nodes and 16 chips with tp8 pp4:

|  Device  |    Model    | Tokens Throughput (tokens/s/p) | Single Step Iteration Time (s/step) |
| :-------: | :----------: | :------------------------------:|:-----------------------------------:|
|   NPUs   | Mixtral-8x7B |             1568.3               |                20.9                |
| Reference | Mixtral-8x7B |              1766.6               |                18.5               |