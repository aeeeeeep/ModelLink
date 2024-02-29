# Qwen

<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/ModelLink/blob/modellink/examples/qwen/README_en.md">English</a> </b> 
</p>

# 目录

- [Qwen-7B](#Qwen-7B)
  - [训练](#训练)
  - [脚本](#脚本)
  - [性能](#性能)
    - [吞吐](#吞吐)

- [Qwen-14B](#Qwen-14B)
  - [训练](#训练)
  - [脚本](#脚本)
  - [性能](#性能)
    - [吞吐](#吞吐)

- [Qwen-72B](#Qwen-72B)
  - [训练](#训练)
  - [脚本](#脚本)
  - [性能](#性能)
    - [吞吐](#吞吐)

# Qwen-7B

## 训练

Qwen-7B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 拷贝代码仓到本地服务器

   ```shell
   git clone -b modellink https://gitee.com/ascend/ModelLink.git 
   cd ModelLink
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

   # 安装加速库
   git clone https://gitee.com/ascend/AscendSpeed.git
   cd AscendSpeed
   pip install -r requirements.txt
   pip3 install -e .
   cd ..
   
   # 安装其余依赖库
   pip install -r requirements.txt
   ```
3. 下载 Qwen-7B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen-7B/tree/main)

   ```bash
   mkdir -p qwen-7b-hf
   cd qwen-7b-hf
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/cache_autogptq_cuda_256.cpp
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/cache_autogptq_cuda_kernel_256.cu
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/configuration_qwen.py
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/cpp_kernels.py
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00001-of-00008.safetensors
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00002-of-00008.safetensors
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00003-of-00008.safetensors
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00004-of-00008.safetensors
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00005-of-00008.safetensors
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00006-of-00008.safetensors
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00007-of-00008.safetensors
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00008-of-00008.safetensors
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/modeling_qwen.py
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/qwen.tiktoken
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/qwen_generation_utils.py
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/tokenization_qwen.py
   wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/tokenizer_config.json

   cd ..
   ```
4. 准备数据集

   下载 Qwen-7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
    # 下载数据
    mkdir dataset_qwen-7b
    cd ./dataset_qwen-7b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
     
    # 处理数据                           
    python ./tools/preprocess_data.py \
        --input ../dataset_qwen-7b/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ../qwen-7b-hf \
        --output-prefix ../dataset_qwen-7b/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 8192 \
        --workers 4 \
        --log-interval 1000 \
    
    cd .. 
   ```
5. 微调

    配置Qwen-7B 微调脚本: examples/qwen/pretrain_qwen_7b_ptd.sh

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-7b-hf/"  #词表路径
    DATA_PATH="./dataset_qwen-7b/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="./qwen-7b-mt/"
   ```

   启动 Qwen-7B 微调脚本: examples/qwen/pretrain_qwen_7b_ptd.sh

   ```shell
    bash examples/qwen/pretrain_qwen_7b_ptd.sh
   ```


### 性能

#### 吞吐

Qwen-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|  设备  |   模型    | tokens吞吐 (tokens/s/p) |
|:----:|:-------:|:---------------------:|
| NPUs | Qwen-7B |         2499          |
|  参考  | Qwen-7B |         2867          |


# Qwen-14B

## 训练

Qwen-14B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 拷贝代码仓到本地服务器

   ```shell
   git clone -b modellink https://gitee.com/ascend/ModelLink.git 
   cd ModelLink 
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

   # 安装加速库
   git clone https://gitee.com/ascend/AscendSpeed.git
   cd AscendSpeed
   pip install -r requirements.txt
   pip3 install -e .
   cd ..
   
   # 安装其余依赖库
   pip install -r requirements.txt
   ```
3. 下载 Qwen-14B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen-14B/tree/main)

   ```bash
   mkdir -p qwen-14b-hf
   cd qwen-14b-hf
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/cache_autogptq_cuda_256.cpp
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/cache_autogptq_cuda_kernel_256.cu
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/configuration_qwen.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/cpp_kernels.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00001-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00002-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00003-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00004-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00005-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00006-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00007-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00008-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00009-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00010-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00011-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00012-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00013-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00014-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00015-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/modeling_qwen.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/qwen.tiktoken
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/qwen_generation_utils.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/tokenization_qwen.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/tokenizer_config.json

   cd ..
   ```
4. 准备数据集

   下载 Qwen-14B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
    # 下载数据
    mkdir dataset_qwen-14b
    cd ./dataset_qwen-14b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
     
    # 处理数据                           
    python ./tools/preprocess_data.py \
        --input ../dataset_qwen-14b/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ../qwen-14b-hf \
        --output-prefix ../dataset_qwen-14b/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 2048 \
        --workers 4 \
        --log-interval 1000 \
    
    cd .. 
   ```
5. 微调

    配置Qwen-14B 微调脚本: examples/qwen/pretrain_qwen_14b_ptd.sh

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-14b-hf/"  #词表路径
    DATA_PATH="./dataset_qwen-14b/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="./qwen-14b-mt/"
   ```

   启动 Qwen-14B 微调脚本: examples/qwen/pretrain_qwen_14b_ptd.sh

   ```shell
    bash examples/qwen/pretrain_qwen_14b_ptd.sh
   ```

### 性能

#### 吞吐

Qwen-14B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|  设备   |    模型    | tokens吞吐 (tokens/s/p) |
|:-----:|:--------:|:---------------------:|
| NPUs | Qwen-14B |         1560          |
|  参考   | Qwen-14B |         1578          |


# Qwen-72B

## 训练

Qwen-72B 训练的硬件配置:

| 硬件 |        配置         |
| :--: |:-----------------:|
| NPU | 128 x Ascend NPUs |

### 脚本

1. 拷贝代码仓到本地服务器

   ```shell
   git clone -b modellink https://gitee.com/ascend/ModelLink.git 
   cd ModelLink
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

   # 安装加速库
   git clone https://gitee.com/ascend/AscendSpeed.git
   cd AscendSpeed
   pip install -r requirements.txt
   pip3 install -e .
   cd ..
   
   # 安装其余依赖库
   pip install -r requirements.txt
   ```
3. 下载 Qwen-72B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen-72B/tree/main)

   ```bash
   mkdir -p qwen-72b-hf
   cd qwen-72b-hf
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/cache_autogptq_cuda_256.cpp
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/cache_autogptq_cuda_kernel_256.cu
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/configuration_qwen.py
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/cpp_kernels.py
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/model-00001-of-000082.safetensors
   ...
   cd ..
   ```
4. 准备数据集

   下载 Qwen-72B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
    # 下载数据
    mkdir dataset_qwen-72b
    cd ./dataset_qwen-72b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
     
    # 处理数据                           
    python ./tools/preprocess_data.py \
        --input ../dataset_qwen-72b/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ../qwen-72b-hf \
        --output-prefix ../dataset_qwen-72b/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 32768 \
        --workers 4 \
        --log-interval 1000 \
    
    cd .. 
   ```
5. 微调

    配置Qwen-72B 微调脚本: examples/qwen/pretrain_qwen_72b_ptd.sh

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-72b-hf/"  #词表路径
    DATA_PATH="./dataset_qwen-72b/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="./qwen-72b-mt/"
   ```

   启动 Qwen-72B 微调脚本: examples/qwen/pretrain_qwen_72b_ptd.sh

   ```shell
    bash examples/qwen/pretrain_qwen_72b_ptd.sh
   ```

### 性能

#### 吞吐

Qwen-72B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|  设备  |    模型    | tokens吞吐 (tokens/s/p)(8k序列) | tokens吞吐 (tokens/s/p)(32k序列) |
|:----:|:--------:|:-----------------------:|:-----------------------:|
| NPUs | Qwen-72B |           285           |           --            |
|  参考  | Qwen-72B |           345           |           --            |