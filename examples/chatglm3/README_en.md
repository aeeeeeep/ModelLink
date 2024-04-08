# ChatGLM
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [ChatGLM3](#ChatGLM3)
- [Contents](#contents)
- [ChatGLM3-6B](#ChatGLM3-6b)
  - [Training](#training)
    - [Script](#script)
    - [Performance](#performance)
      - [Machine performance](#machine-performance)
  - [Inference-6B](#inference-6b)
  - [Evaluation-6B](#evaluation-6b)

# ChatGLM3-6B

## Training

Here's a hardware summary of pre-training  ChatGLM3-6B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git 
    cd ModelLink
    mkdir logs
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    ```
2. Build environment

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # install torch and torch_npu
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # modify ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # install AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```
3. Prepare pretrained weights and tokenizer
    Download the ChatGLM3-6B checkpoint from [here](https://huggingface.co/THUDM/chatglm3-6b/tree/main)

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
4. weight conversion in ptd mode

   4.1 Convert weights from HuggingFace format to Megatron format 
   ***（This scenario is generally used to enable the open-source HuggingFace model to be trained on Megatron）***

   ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # convert to ptd weights
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader chatglm3_hf \
        --saver megatron \
        --target-tensor-parallel-size 2 \
        --target-pipeline-parallel-size 2 \
        --load-dir ./model_from_hf/chatglm3-6b-hf/ \
        --save-dir ./model_weights/chatglm3-6b-tp2-pp2/ \
        --tokenizer-model ./model_from_hf/chatglm3-6b-hf/tokenizer.model
   ```

    Note: The --target-tensor-parallel-size of chatglm3 is related to the multi_query_attention configuration in the config.json, and the multi_query_attention set here is 2.

