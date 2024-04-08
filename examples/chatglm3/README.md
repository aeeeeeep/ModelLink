# ChatGLM3

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [ChatGLM3](#ChatGLM3)
- [目录](#目录)
- [ChatGLM3-6B](#ChatGLM3-6B)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
  - [推理-6B](#推理-6B)
  - [评估-6B](#评估-6B)

# ChatGLM3-6B

## 训练

ChatGLM3-6B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器

   ```shell
   git clone https://gitee.com/ascend/ModelLink.git 
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
   pip install -r requirements.txt 
   pip3 install -e .
   cd ..

   # 安装其余依赖库
   pip install -r requirements.txt 
   ```
3. 下载 ChatGLM3-6B 的 [预训练权重和词表](https://huggingface.co/THUDM/chatglm3-6b/tree/main)

   ```shell
     #!/bin/bash
     mkdir ./model_from_hf/chatglm3-6b-hf/
     cd ./model_from_hf/chatglm3-6b-hf/
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/config.json
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/configuration_chatglm.py
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/pytorch_model-00001-of-00007.bin
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/pytorch_model-00002-of-00007.bin
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/pytorch_model-00003-of-00007.bin
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/pytorch_model-00004-of-00007.bin
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/pytorch_model-00005-of-00007.bin
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/pytorch_model-00006-of-00007.bin
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/pytorch_model-00007-of-00007.bin
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/pytorch_model.bin.index.json
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/quantization.py
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenizer.model
     wget https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenizer_config.json
     cd ../../
   ```
4. 权重转换
   4.1 将权重从 huggingface 格式转化为 megatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader chatglm3_hf \
        --saver megatron \
        --target-tensor-parallel-size 2 \
        --target-pipeline-parallel-size 2 \
        --load-dir ./model_from_hf/chatglm3-6b-hf/ \
        --save-dir ./model_weights/chatglm3-6b-tp2-pp2/ \
        --tokenizer-model ./model_from_hf/chatglm3-6b-hf/tokenizer.model \
        --add-qkv-bias
   ```

   注意：chatglm3的--target-tensor-parallel-size跟config.json中的multi_query_attention配置有关，这里multi_query_attention设置的是2。