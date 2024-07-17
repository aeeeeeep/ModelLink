# Modellink模型训练

<p align="left">
        <b>简体中文</b> 
</p>

# 目录

- [Modellink模型训练](#Modellink模型训练)
- [目录](#目录)
- [环境安装](#环境安装)
- [预训练](#预训练)

# 环境安装


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
3. 从[Huggingface](https://huggingface.co/)下载模型的**预训练权重**、**词表**
    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/模型文件夹/
    cd ./model_from_hf/模型文件夹/
    wget https://Huggingface预训练权重路径
    wget https://Huggingface词表路径
    ...
    cd ../../
    ```
    **示例：** *(以grok1-40B为例)*

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/grok1-40B/
    cd ./model_from_hf/grok1-40B/
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/config.json
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/configuration_grok1.py
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/modeling_grok1.py
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/modeling_grok1_outputs.py
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/special_tokens_map.json
    wget https://huggingface.co/hpcai-tech/grok-1/resolve/main/tokenizer_config.json
    wget wget https://github.com/xai-org/grok-1/raw/main/tokenizer.model
    cd ../../
    ```

# 预训练

4. 准备数据集

    4.1 下载[数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)   

    ```shell

    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    4.2 处理数据
    ```shell
    mkdir ./dataset/模型文件夹/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/模型文件夹/ \ # <--tokenizer路径
        --output-prefix ./dataset/模型文件夹/数据集文件前缀 \
        --workers worker数量 \
        --log-interval 日志记录间隔 \
        --tokenizer-type PretrainedFromHF
    ```
    **示例：** *(以grok1-40B为例)*
    ```shell
    mkdir ./dataset/grok1-40B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/grok1-40B/ \
        --output-prefix ./dataset/grok1-40B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

5. 启动预训练
    5.1 在example中找到模型预训练脚本: pretrain_xxx_xx.sh
    5.2 配置参数,根据实际情况修改
    
    **示例：** *(以grok1-40B为例)*

    ```shell
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/grok1-40B/"  #模型参数保存路径
    TOKENIZER_MODEL="./model_from_hf/grok1-40B/tokenizer.model"  #词表路径
    DATA_PATH="./dataset/grok1-40B/alpaca_text_document"  #数据集路径

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
    多机运行增加参数--overlap-grad-reduce

    5.3 启动预训练脚本

    ```shell
    bash pretrain_xxx_xxx.sh
    ```
    **示例：** *(以grok1-40B为例)*
    ```shell
    bash examples/grok1/pretrain_grok1_40b_ptd.sh
    ```
    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。
