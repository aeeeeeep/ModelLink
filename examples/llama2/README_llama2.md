# LLaMA2

<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/AscendSpeed/blob/master/examples/llama2/README_en.md">English</a> </b> 
</p>

# 目录

- [LLAMA2-7B](#LLAMA2-7B)
  - [训练](#训练)
  - [脚本](#脚本)
  - [性能](#性能)
    - [吞吐](#吞吐)
    - [精度](#精度)



# LLAMA2-7B

## 训练

LLAMA2-7B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 拷贝代码仓到本地服务器

   ```shell
   git clone https://gitee.com/ascend/AscendSpeed.git 
   cd AscendSpeed 
   git checkout modellink
   pip install -r requirements.txt
   cd ..
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
   
   # 安装 megatron
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cp -r Megatron-LM/megatron ./AscendSpeed
   
   # 安装加速库
   git clone https://gitee.com/ascend/AscendSpeed2.git
   cd AscendSpeed2
   pip3 install -e .
   ```
3. 下载 LLAMA2-7B 的 [预训练权重和词表](https://huggingface.co/daryl149/llama-2-7b-hf/tree/main)

   ```shell
     #!/bin/bash
     mkdir -p llama-2-7b-hf
     cd llama-2-7b-hf
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/config.json
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/generation_config.json
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00002-of-00002.bin
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model.bin.index.json
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/special_tokens_map.json
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
     wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
     cd ..
   ```

   将权重从 huggingface 格式转化为 magatron 格式 

   ```bash
    cd AscendSpeed
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
    # 权重格式转换
    python tools/checkpoint/util.py --model-type GPT \
                                    --loader llama2_hf \
                                    --saver megatron \
                                    --target-tensor-parallel-size 8 \
                                    --load-dir {your huggingface ckpt path} \
                                    --save-dir {your megatron ckpt save path} \
                                    --tokenizer-model {your load tokenizer-model path}
   cd ..
   ```
4. 预训练


   4.1 准备数据集

   下载 LLaMA2-7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
     # 下载数据
     mkdir dataset_llama2
     cd ./dataset_llama2
     wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     cd AscendSpeed
     # 处理数据                           
     python ./tools/preprocess_data.py \
       --input ../dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ../llama-2-7b-hf \
       --output-prefix ../dataset_llama2/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF
    cd .. 
   ```
   4.2 预训练
   ```shell
    # 配置LLaMA2-7B 预训练脚本: pretrain_llama2_7b.sh
    cd AscendSpeed
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_PATH=./llama-2-7b-hf/  #词表路径
    DATA_PATH=./dataset_llama2/alpaca_text_document  #数据集路径
   ```

   启动 LLaMA2-7B 预训练脚本: examples/pretrain_llama2_7b.sh

   ```shell
    bash examples/llama2/pretrain_llama2_7b.sh 
   ```
5. 微调

   微调的配置脚本基本和预训练脚本pretrain_llama2_7b.sh一致. *区别是增加权重路径参数*
```shell
  # 根据实际情况配置模型参数加载路径
  CKPT_LOAD_DIR="your init model load path"
```

### 性能

#### 吞吐

LLaMA2-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 |   模型   | 迭代数 | 样本吞吐 (samples/step) | tokens吞吐 (tokens/s/p) | 单步迭代时间 (s/step) | 浮点计算数 (TFLOPs/s) |
| :--: | :-------: | :----: | :--------------------: | :---------------------: | :-------------------: | :-------------------: |
| NPUs | LLaMA2-7B |  1024  |         4.99         |        2562        |         3.2         |        117.8        |
| 参考 | LLaMA2-7B |  1024  |         5.63         |         2884         |         2.84         |        131.96        |

#### 精度

5000步的均方误差为0.00000195
