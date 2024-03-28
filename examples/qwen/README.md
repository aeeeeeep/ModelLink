# Qwen

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [Qwen-7B](#Qwen-7B)
  - [训练](#训练)
  - [脚本](#脚本)
  - [性能](#性能)
    - [吞吐](#吞吐)
  - [推理](#推理)
  - [评估](#评估)

- [Qwen-14B](#Qwen-14B)
  - [训练](#训练)
  - [脚本](#脚本)
  - [性能](#性能)
    - [吞吐](#吞吐)
  - [推理](#推理)
  - [评估](#评估)

- [Qwen-72B](#Qwen-72B)
  - [训练](#训练)
  - [脚本](#脚本)
  - [性能](#性能)
    - [吞吐](#吞吐)
  - [推理](#推理)
  - [评估](#评估)

# Qwen-7B

## 训练

Qwen-7B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 拷贝代码仓到本地服务器
   ```shell
   git clone https://gitee.com/ascend/ModelLink.git 
   cd ModelLink
   mkdir logs
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
   pip install -e .
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
   ```
   修改modelling_qwen.py文件第39行，将：
   ```python
   SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
   ```
   修改为：
   ```python
   SUPPORT_FP16 = True
   ```
   再执行：
   ```bash
   cd ..
   ```
4. 权重转换

   将权重从 huggingface 格式转化为 megatron 格式
  ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```shell
   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
   python tools/checkpoint/util.py --model-type GPT \
                                   --loader qwen_hf \
                                   --saver megatron \
                                   --target-tensor-parallel-size 8 \
                                   --load-dir ./qwen-7b-hf \
                                   --save-dir {your megatron ckpt save path} \
                                   --tokenizer-model ./qwen-7b-hf/qwen.tiktoken \
                                   --add-qkv-bias
   ```

   任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
  ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***
   
   ```bash
   cd ModelLink/
   # 请按照您的真实环境修改 set_env.sh 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python tools/checkpoint/util.py --model-type GPT \
                                   --loader megatron \
                                   --saver megatron \
                                   --save-model-type save_huggingface_qwen \
                                   --load-dir {your model ckpt save path} \
                                   --target-tensor-parallel-size 1 \
                                   --target-pipeline-parallel-size 1 \
                                   --add-qkv-bias \
                                   --save-dir ./qwen-7b-hf     # 需要填入原始HF模型路径，新权重会存于./qwen-7b-hf/mg2hg
   ```

5. 准备数据集

   下载 Qwen-7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
    # 下载数据
    mkdir dataset_qwen-7b
    cd ./dataset_qwen-7b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
     
    # 处理数据                           
    python ./tools/preprocess_data.py \
        --input ./dataset_qwen-7b/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./qwen-7b-hf \
        --output-prefix ./dataset_qwen-7b/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 8192 \
        --workers 4 \
        --log-interval 1000
   ```
   
6. 预训练
    配置Qwen-7B 预训练脚本: examples/qwen/pretrain_qwen_7b_ptd.sh

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-7b-hf/"  #词表路径
    DATA_PATH="./dataset_qwen-7b/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="your megatron ckpt save path"
   ```

   启动 Qwen-7B 预训练脚本: examples/qwen/pretrain_qwen_7b_ptd.sh

   ```shell
    bash examples/qwen/pretrain_qwen_7b_ptd.sh
   ```
   **注意**：如果使用多机训练，需要设置多机数据共享，非主节点通过数据共享读取主节点数据。或者，直接将主节点生成的数据复制到非主节点。


### 性能

#### 吞吐

Qwen-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|  设备  |   模型    | tokens吞吐 (tokens/s/p) |
|:----:|:-------:|:---------------------:|
| NPUs | Qwen-7B |         2499          |
|  参考  | Qwen-7B |         2867          |


## 推理

配置 qwen-7b 推理脚本：tasks/inference/generate_qwen_7b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 修改模型权重路径和词表路径
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
```

启动qwen-7b推理脚本
```bash
bash tasks/inference/generate_qwen_7b_ptd.sh
```

推理示例如下：
![Inference](../../sources/images/qwen/qwen_7b_inference.png)


## 评估

使用[CEval数据集](https://huggingface.co/datasets/ceval/ceval-exam)和[MMLU数据集](https://huggingface.co/datasets/cais/mmlu)评估模型.

配置qwen-7b评估脚本: tasks/evaluation/evaluate_qwen_7b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH=./qwen-7b-hf  #词表路径
CHECKPOINT="your model directory path"  #模型路径

# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"  # ceval任务配置为 "./ceval/val/"
TASK="mmlu"  # ceval任务配置为 "ceval"
```

启动评估

```bash
bash tasks/evaluation/evaluate_qwen_7b_ptd.sh
```

| 数据集 | 总学科数 | 总问题数 |                                     参考准确率                                     | NPU准确率 |
|:---:|:---:|:---:|:-----------------------------------------------------------------------------:|:---:|
| CEval | 52 | 1346 |                  [63.5](https://huggingface.co/Qwen/Qwen-7B)                  | 62.5 |
| MMLU | 57 | 14042 |                  [58.2](https://huggingface.co/Qwen/Qwen-7B)                  | 58.1 |

# Qwen-14B

## 训练

Qwen-14B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 拷贝代码仓到本地服务器

   ```shell
   git clone https://gitee.com/ascend/ModelLink.git 
   cd ModelLink
   mkdir logs 
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
   pip install -e .
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
   ```
   修改modelling_qwen.py文件第39行，将：
   ```python
   SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
   ```
   修改为：
   ```python
   SUPPORT_FP16 = True
   ```
   再执行：
   ```bash
   cd ..
   ```
4. 权重转换

   将权重从 huggingface 格式转化为 megatron 格式
  ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```bash
   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
   python tools/checkpoint/util.py --model-type GPT \
                                   --loader qwen_hf \
                                   --saver megatron \
                                   --target-tensor-parallel-size 8 \
                                   --load-dir ./qwen-14b-hf \
                                   --save-dir {your megatron ckpt save path} \
                                   --tokenizer-model ./qwen-14b-hf/qwen.tiktoken \
                                   --add-qkv-bias
   ```
   任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
  ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***
   ```shell
   cd ModelLink/
   # 请按照您的真实环境修改 set_env.sh 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python tools/checkpoint/util.py --model-type GPT \
      --loader megatron \
      --saver megatron \
      --save-model-type save_huggingface_qwen \
      --load-dir {your model ckpt save path} \
      --target-tensor-parallel-size 1 \
      --target-pipeline-parallel-size 1 \
      --add-qkv-bias \
      --save-dir ./qwen-14b-hf     # 需要填入原始HF模型路径，新权重会存于./qwen-14b-hf/mg2hg
   ```

5. 准备数据集

   下载 Qwen-14B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
    # 下载数据
    mkdir dataset_qwen-14b
    cd ./dataset_qwen-14b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
     
    # 处理数据                           
    python ./tools/preprocess_data.py \
        --input ./dataset_qwen-14b/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./qwen-14b-hf \
        --output-prefix ./dataset_qwen-14b/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 2048 \
        --workers 4 \
        --log-interval 1000
   ```
6. 预训练

    配置Qwen-14B 预训练脚本: examples/qwen/pretrain_qwen_14b_ptd.sh

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-14b-hf/"  #词表路径
    DATA_PATH="./dataset_qwen-14b/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="your megatron ckpt save path"
   ```

   启动 Qwen-14B 预训练脚本: examples/qwen/pretrain_qwen_14b_ptd.sh

   ```shell
    bash examples/qwen/pretrain_qwen_14b_ptd.sh
   ```
    **注意**：如果使用多机训练，需要设置多机数据共享，非主节点通过数据共享读取主节点数据。或者，直接将主节点生成的数据复制到非主节点。
### 性能

#### 吞吐

Qwen-14B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|  设备   |    模型    | tokens吞吐 (tokens/s/p) |
|:-----:|:--------:|:---------------------:|
| NPUs | Qwen-14B |         1560          |
|  参考   | Qwen-14B |         1578          |


## 推理

配置 qwen-14b 推理脚本：tasks/inference/generate_qwen_14b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 修改模型权重路径和词表路径
CHECKPOINT="your model directory path"
TOKENIZER_PATH=./qwen-14b-hf
```

启动qwen-14b推理脚本
```bash
bash tasks/inference/generate_qwen_14b_ptd.sh
```

推理示例如下：
![Inference](../../sources/images/qwen/qwen_14b_inference.png)


## 评估

使用[CEval数据集](https://huggingface.co/datasets/ceval/ceval-exam)和[MMLU数据集](https://huggingface.co/datasets/cais/mmlu)评估模型.

配置qwen-14b评估脚本: tasks/evaluation/evaluate_qwen_14b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH=./qwen-14b-hf  #词表路径
CHECKPOINT="your model directory path"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"  # ceval任务配置为 "./ceval/val/"
TASK="mmlu"  # ceval任务配置为 "ceval"
```

启动评估

```bash
bash tasks/evaluation/evaluate_qwen_14b_ptd.sh
```

| 数据集 | 总学科数 | 总问题数 |                    参考准确率                     | NPU准确率 |
|:---:|:---:|:---:|:--------------------------------------------:|:---:|
| CEval | 52 | 1346 | [72.1](https://huggingface.co/Qwen/Qwen-14B) | 71.1 |
| MMLU | 57 | 14042 | [66.3](https://huggingface.co/Qwen/Qwen-14B) | 65.3 |


# Qwen-72B

## 训练

Qwen-72B 训练的硬件配置:

| 硬件  | 序列长度 |        配置         |
|:---:|:----:|:-----------------:|
| NPU |  8k  | 64 x Ascend NPUs  |
| NPU | 32k  | 320 x Ascend NPUs |

### 脚本

1. 拷贝代码仓到本地服务器

   ```shell
   git clone https://gitee.com/ascend/ModelLink.git 
   cd ModelLink
   mkdir logs
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
   pip install -e .
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
   ```
   修改modelling_qwen.py文件第39行，将：
   ```python
   SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
   ```
   修改为：
   ```python
   SUPPORT_FP16 = True
   ```
   再执行：
   ```bash
   cd ..
   ```
4. 权重转换

   将权重从 huggingface 格式转化为 megatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```bash
   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
      
   python tools/checkpoint/util.py --model-type GPT \
                                   --loader qwen_hf \
                                   --saver megatron \
                                   --target-tensor-parallel-size 8 \
                                   --load-dir ./qwen-72b-hf \
                                   --save-dir {your megatron ckpt save path} \
                                   --tokenizer-model ./qwen-72b-hf/qwen.tiktoken \
                                   --add-qkv-bias
   ```
   
   任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
   ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***
   ```shell
   cd ModelLink/
   # 请按照您的真实环境修改 set_env.sh 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python tools/checkpoint/util.py --model-type GPT \
      --loader megatron \
      --saver megatron \
      --save-model-type save_huggingface_qwen \
      --load-dir {your model ckpt save path} \
      --target-tensor-parallel-size 1 \
      --target-pipeline-parallel-size 1 \
      --add-qkv-bias \
      --save-dir ./qwen-72b-hf     # 需要填入原始HF模型路径，新权重会存于./qwen-72b-hf/mg2hg
   ```

5. 准备数据集

   下载 Qwen-72B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)
   ```shell
    # 下载数据
    mkdir dataset_qwen-72b
    cd ./dataset_qwen-72b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
     
    # 处理数据                           
    python ./tools/preprocess_data.py \
        --input ./dataset_qwen-72b/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./qwen-72b-hf \
        --output-prefix ./dataset_qwen-72b/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 8192 \
        --workers 4 \
        --log-interval 1000
   ```
   
6. 预训练

    配置Qwen-72B 预训练脚本: examples/qwen/pretrain_qwen_72b_ptd.sh

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-72b-hf/"  #词表路径
    DATA_PATH="./dataset_qwen-72b/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="your megatron ckpt save path"
   ```
   
   若使用32k长序列，需要开启重计算特性并修改seq-length参数值为32768，参数配置如下：

   ```shell
   --seq-length 32768 \

    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 80 \
   ```

   启动 Qwen-72B 预训练脚本: examples/qwen/pretrain_qwen_72b_ptd.sh

   ```shell
    bash examples/qwen/pretrain_qwen_72b_ptd.sh
   ```
    **注意**：如果使用多机训练，需要设置多机数据共享，非主节点通过数据共享读取主节点数据。或者，直接将主节点生成的数据复制到非主节点。
### 性能

#### 吞吐

Qwen-72B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|  设备  |    模型    | tokens吞吐 (tokens/s/p)(8k序列) | tokens吞吐 (tokens/s/p)(32k序列) |
|:----:|:--------:|:-----------------------:|:-----------------------:|
| NPUs | Qwen-72B |           285           |           --            |
|  参考  | Qwen-72B |           345           |           --            |


## 推理

配置 qwen-72b 推理脚本：tasks/inference/generate_qwen_72b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 修改模型权重路径和词表路径
CHECKPOINT="your model directory path"
TOKENIZER_PATH=./qwen-72b-hf
```

启动qwen-72b推理脚本
```bash
bash tasks/inference/generate_qwen_72b_ptd.sh
```

推理示例如下：
![Inference](../../sources/images/qwen/qwen_72b_inference.png)


## 评估

使用[CEval数据集](https://huggingface.co/datasets/ceval/ceval-exam)和[MMLU数据集](https://huggingface.co/datasets/cais/mmlu)评估模型.

配置qwen-72b评估脚本: tasks/evaluation/evaluate_qwen_72b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH=./qwen-72b-hf  #词表路径
CHECKPOINT="your model directory path"  #模型路径

# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"  # ceval任务配置为 "./ceval/val/"
TASK="mmlu"  # ceval任务配置为 "ceval"
```

启动评估

```bash
bash tasks/evaluation/evaluate_qwen_72b_ptd.sh
```

| 数据集 | 总学科数 | 总问题数 |                    参考准确率                     | NPU准确率 |
|:---:|:---:|:---:|:--------------------------------------------:|:---:|
| CEval | 52 | 1346 | [83.3](https://huggingface.co/Qwen/Qwen-72B) | 81.8 |
| MMLU | 57 | 14042 | [77.4](https://huggingface.co/Qwen/Qwen-72B) | 74.6 |