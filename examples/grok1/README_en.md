# Grok1 $\color{black}{\rm\tiny{【Model}}$ $\color{black}{\rm\tiny{contributed}}$ $\color{black}{\rm\tiny{by}}$ $\color{black}{\rm\tiny{Ascend】}}$ 

<p align="left">
        <b><a href="README.md">简体中文</a> </b> |
        <b>English</b> 
</p>

# Table of Contents

- [Grok1](#Grok1)
- [Table of Contents](#table-of-contents)
- [40B](#40B)
  - [Hardware-Requirements](#hardware-requirements)
  - [Preparation](#preparation)
  - [Model-Training](#model-training)
  - [Model-Performance](#model-performance)
    - [Throughput](#throughput)

# 40B

## Hardware-Requirements

Minimum hardware requirements for training:

| Hardware |  Configuration   |
| :------: |:----------------:|
|   NPU   | 16 x Ascend NPUs |

Recommended hardware configuration for inference:

| Hardware |  Configuration   |
| :------: |:----------------:|
|   NPU   | 16 x Ascend NPUs |

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
    mkdir model_from_hf/grok1-40B
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
    pip install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```

3. Download the model files and vocabulary for Grok1-40B from [here](https://huggingface.co/hpcai-tech/grok-1/tree/main) and [tokenizer.model](https://github.com/xai-org/grok-1/blob/main/tokenizer.model)

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

## Model-Training

1. Prepare dataset

    Download the datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet), save to ModelLink/dataset/ directory.

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # process datasets
    mkdir ./dataset/grok1-40B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/grok1-40B/ \
        --output-prefix ./dataset/grok1-40B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    Configure grok1-40B pre-training script: ***examples/grok1/pretrain_grok1_40b_ptd.sh***

    ```shell
    # Set the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # Configure according to the actual vocabulary, dataset, and model parameter save path
    DATA_PATH="./dataset/grok1-40B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/grok1-40B/tokenizer.model"
    CKPT_SAVE_DIR="./ckpt/grok1-40B/"

    # Configure distributed parameters according to the actual distributed cluster
    GPUS_PER_NODE=8
    MASTER_ADDR="your master node IP"
    MASTER_PORT=6000
    NNODES=2
    NODE_RANK="current node id"
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

    # Training parallel strategy
    TP=4
    PP=2
    EP=2
    ```

    Start grok1-40B pre-training script: ***examples/grok1/pretrain_grok1_40b_ptd.sh***

    ```shell
    bash examples/grok1/pretrain_grok1_40b_ptd.sh
    ```

    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

## Model-Performance

### Throughput

Comparison of Mixtral-8x7B performance on 4 nodes and 32 chips with tp8 pp4:

|  Device  |    Model    | Tokens Throughput (tokens/s/p) | Single Step Iteration Time (s/step) |
| :-------: | :----------: |:------------------------------:|:-----------------------------------:|
|   NPUs   | grok1-40B | 1646.633               |                39.8                 |
| Reference | grok1-40B |2057               |                 32                  |
