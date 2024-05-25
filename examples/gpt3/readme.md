# GPT3  $\color{black}{\bf\tiny{【社区贡献模型】}}$

<p align="left">
        <b>简体中文</b> |
        <b><a href="README.md">English</a> </b> 
</p>

# 目录

- [GPT3](#GPT3)
- [目录](#目录)
- [GPT3-175B](#GPT3-175B)
  - [训练-175B](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)

# GPT3-175B

## 训练

GPT3-175B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 128 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器:

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout -f bcce6f
    cp -r megatron ../ModelLink/
    cd .. 
    cd ModelLink
    mkdir logs
    mkdir vocab_file
    mkdir dataset
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

    # 安装 AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其他依赖
    pip install -r requirements.txt 
    ```

3. 准备数据、词表来拉起模型
    3.1 准备数据
    
    可以从 [这里](https://huggingface.co/datasets/wikipedia/tree/main/data/20220301.en) 下载原始数据
    ```shell
    # 下载 enwiki 数据
    # 总共有 41 个文件，我们可以选择部分来制作数据
    cd ./dataset
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00000-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00001-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00002-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00003-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00004-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00005-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00006-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00007-of-00041.parquet
    cd ..

    # 下载 vocab file 和 merge table
    cd vocab_file
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    cd ..

    # 处理成训练数据
    python ./tools/preprocess_data.py \
        --input ./dataset/ \
        --output-prefix ./dataset/gpt_text_sentence \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file ./vocab_file/gpt2-vocab.json \
        --merge-file ./vocab_file/gpt2-merges.txt \
        --append-eod \
        --workers 4 \
        --log-interval 1000
    ```

    3.2 用 ptd 模式进行预训练
    配置 GPT3-175B PTD 预训练脚本: examples/gpt3/pretrain_gpt3_175B.sh

    ```shell
    # 请根据真实情况配置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # 请根据真实存放路径配置以下参数
    VOCAB_FILE="./vocab_file/gpt2-vocab.json"   # 词表
    MERGE_FILE="./vocab_file/gpt2-merges.txt"   # BPE 合并表
    DATA_PATH="./dataset/gpt_text_sentence"     # 数据路径
    ```

   拉起 GPT3-175B PTD 预训练脚本: examples/gpt3/pretrain_gpt3_175B.sh

    ```shell
    bash examples/gpt3/pretrain_gpt3_175B.sh
    ```

### 性能

#### 吞吐

GPT3-175B 在 **昇腾芯片**上的性能数据：

| 设备 |    模型    | tokens吞吐 (tokens/s/p) | 
| :--: | :--------: |:---------------------:| 
| NPUs | GPT3-175B |        153.1         |

