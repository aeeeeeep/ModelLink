# ChatGLM
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [ChatGLM3](#ChatGLM3)
- [Contents](#contents)
- [ChatGLM3-6B](#ChatGLM3-6b)
  - [Training-6B](#training)
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
3. Prepare pretrained weights and tokenizer
    Download the ChatGLM3-6B checkpoint from [here](https://huggingface.co/THUDM/chatglm3-6b/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/chatglm3_6b_hf/
    cd ./model_from_hf/chatglm3_6b_hf/
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/config.json
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/configuration_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/modeling_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/pytorch_model-00001-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/pytorch_model-00002-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/pytorch_model-00003-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/pytorch_model-00004-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/pytorch_model-00005-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/pytorch_model-00006-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/pytorch_model-00007-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/pytorch_model.bin.index.json
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/quantization.py
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/tokenization_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/tokenizer.model
    wget https://huggingface.co/THUDM/chatglm3-6b/resolve/main/tokenizer_config.json
    cd ../../
    ```
4. weight conversion in ptd mode < Not supported yet >

   
5. pre-training

    5.1 Prepare dataset 

    Download the ChatGLM3-6B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # process datasets  
    mkdir ./dataset/chatglm3_6b_hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/chatglm3_6b_hf/ \
        --output-prefix ./dataset/chatglm3_6b_hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 pre-training using ptd mode

    We support ChatGLM3-6B model pre-training with sequence lengths of 32K and 64K.

    Config ChatGLM3-6B-32K/64K pre-training script:

    ChatGLM3-6B-32K: examples/chatglm3/mcore/pretrain_chatglm3_6B_32K.sh

    ChatGLM3-6B-64K: examples/chatglm3/mcore/pretrain_chatglm3_6B_64K.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    LOAD_CHECKPOINT_PATH="./model_weights/chatglm3_6b_tp1pp2/"
    SAVE_CHECKPOINT_PATH="./ckpt/chatglm3_6b_hf/"
    TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"  #tokenizer path
    DATA_PATH="./dataset/chatglm3_6b_hf/alpaca_text_document"  #processed dataset
    ```

    **Note**: Multi-machine training requires the addition of parameter --overlap-grad-reduce

    Launch ChatGLM3-6B pre-training script: 

    ChatGLM3-6B-32K
    ```shell
    bash examples/chatglm3/mcore/pretrain_chatglm3_6B_32K.sh
    ```

    ChatGLM3-6B-64K
    ```shell
     bash examples/chatglm3/mcore/pretrain_chatglm3_6B_64K.sh
    ```
   
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.


### Performance

#### Machine performance

The performance of ChatGLM3-6B in **Ascend NPU** and **Reference**:

[//]: # ()
[//]: # (|  Device   |      Model      | sequence length | throughput rate &#40;tokens/s/p&#41; | )

[//]: # (| :--: |:---------------:| :--------:|:---------------------:| )

[//]: # (| NPUs | ChatGLM3-6B-8K  | 8192 |       4297        |  )

[//]: # (| Reference | ChatGLM3-6B-8K  |  8192 |      4269         |  )

[//]: # (| NPUs | ChatGLM3-6B-32K | 32768 |       2501        |  )

[//]: # (| Reference | ChatGLM3-6B-32K |  32768 |      4269         |  )

[//]: # (|   NPUs    | ChatGLM3-6B-64K | 65536 |       1665        |  )

[//]: # (| Reference | ChatGLM3-6B-64K |  65536 |      2112         |  )

<table>
    <tr align="center">
        <td></td>
        <td>设备</td>
        <td>ChatGLM3-6B-8K</td>
        <td>ChatGLM3-6B-32K</td>
        <td>ChatGLM3-6B-64K</td>
    </tr>
    <tr align="center">
        <td rowspan="2">tokens吞吐 (tokens/s/p)</td>
        <td>NPUs</td>
        <td>4297</td>
        <td>2501</td>
        <td>1665</td>
    </tr>
    <tr align="center">
        <td>参考</td>
        <td>4269</td>
        <td>2942</td>
        <td>2112</td>
    </tr>
</table>
