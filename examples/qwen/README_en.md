# Qwen
<p align="left">
        <b><a href="https://gitee.com/ascend/ModelLink/blob/modellink/examples/qwen/README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [Qwen-7B](#Qwen-7B)
  - [Training](#training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)

- [Qwen-14B](#Qwen-14B)
  - [Training](#training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)

- [Qwen-72B](#Qwen-72B)
  - [Training](#training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)

# Qwen-7B

## Training

Here's a hardware summary of pre-training  Qwen-7B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:
    ```shell
    git clone -b modellink https://gitee.com/ascend/ModelLink.git 
    cd ModelLink
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
    
    # install AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    pip install -r requirements.txt
    pip install -e .
    cd ..
   
    # install other packages
    pip install -r requirements.txt
    ```

3. Prepare pretrained weights and tokenizer
    Download the Qwen-7B checkpoint from [here](https://huggingface.co/Qwen/Qwen-7B/tree/main) 
    
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
4. Prepare dataset

	Download the Qwen-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)    
	```shell
    # download datasets
    mkdir dataset_qwen-7b
    cd ./dataset_qwen-7b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
	
	# process datasets                              
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

   Convert weights from huggingface format to megatron format

    ```bash
    cd ModelLink
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python tools/checkpoint/util.py --model-type GPT \
                                    --loader qwen_hf \
                                    --saver megatron \
                                    --target-tensor-parallel-size 8 \
                                    --load-dir ../qwen-7b-hf \
                                    --save-dir {your megatron ckpt save path} \
                                    --tokenizer-model ../qwen-7b-hf/qwen.tiktoken \
                                    --add-qkv-bias
    
    cd ..
    ```

5. fine-tuning

	Config Qwen-7B fine-tuning script: examples/qwen/pretrain_qwen_7b_ptd.sh 
   ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-7b-hf/"  #tokenizer path
    DATA_PATH="./dataset_qwen-7b/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="your megatron ckpt save path"
   ```

	Launch Qwen-7B fine-tuning script: examples/qwen/pretrain_qwen_7b_ptd.sh
   
   ```shell
    bash examples/qwen/pretrain_qwen_7b_ptd.sh 
   ```

### Performance

#### Machine performance

The performance of Qwen-7B in **Ascend NPU** and **Reference**:

|  Device   |  Model  | throughput rate (tokens/s/p) |
|:---------:|:-------:|:----------------------------:|
|   NPUs    | Qwen-7B |             2499             |
| Reference | Qwen-7B |             2867             |


# Qwen-14B

## Training

Here's a hardware summary of pre-training  Qwen-14B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:
    ```shell
    git clone -b modellink https://gitee.com/ascend/ModelLink.git 
    cd ModelLink
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
    
    # install AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    pip install -r requirements.txt
    pip install -e .
    cd ..
   
    # install other packages
    pip install -r requirements.txt
    ```

3. Prepare pretrained weights and tokenizer
    Download the Qwen-14B checkpoint from [here](https://huggingface.co/Qwen/Qwen-14B/tree/main) 
    
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

   Convert weights from huggingface format to megatron format

    ```bash
    cd ModelLink
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python tools/checkpoint/util.py --model-type GPT \
                                    --loader qwen_hf \
                                    --saver megatron \
                                    --target-tensor-parallel-size 8 \
                                    --load-dir ../qwen-14b-hf \
                                    --save-dir {your megatron ckpt save path} \
                                    --tokenizer-model ../qwen-14b-hf/qwen.tiktoken \
                                    --add-qkv-bias
    
    cd ..
    ```

4. Prepare dataset

	Download the Qwen-14B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)    
	```shell
    # download datasets
    mkdir dataset_qwen-14b
    cd ./dataset_qwen-14b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
	
	# process datasets                              
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

5. fine-tuning

	Config Qwen-14B fine-tuning script: examples/qwen/pretrain_qwen_14b_ptd.sh 
   ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-14b-hf/"  #tokenizer path
    DATA_PATH="./dataset_qwen-14b/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="your megatron ckpt save path"
   ```

	Launch Qwen-14B fine-tuning script: examples/qwen/pretrain_qwen_14b_ptd.sh
   
   ```shell
    bash examples/qwen/pretrain_qwen_14b_ptd.sh 
   ```

### Performance

#### Machine performance

The performance of Qwen-14B in **Ascend NPU** and **Reference**:

|  Device   |  Model   | throughput rate (tokens/s/p) |
|:---------:|:--------:|:----------------------------:|
|   NPUs    | Qwen-14B |             1560             |
| Reference | Qwen-14B |             1578             |


# Qwen-72B

## Training

Here's a hardware summary of pre-training  Qwen-72B:

| Hardware |       Value       |
| :------: |:-----------------:|
|   NPU    | 128 x Ascend NPUs |

### Script

1. Clone the repository to your local server:
    ```shell
    git clone -b modellink https://gitee.com/ascend/ModelLink.git 
    cd ModelLink
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
    
    # install AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    pip install -r requirements.txt
    pip install -e .
    cd ..
   
    # install other packages
    pip install -r requirements.txt
    ```

3. Prepare pretrained weights and tokenizer
    Download the Qwen-72B checkpoint from [here](https://huggingface.co/Qwen/Qwen-72B/tree/main) 
    
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
   
   Convert weights from huggingface format to megatron format

    ```bash
    cd ModelLink
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python tools/checkpoint/util.py --model-type GPT \
                                    --loader qwen_hf \
                                    --saver megatron \
                                    --target-tensor-parallel-size 8 \
                                    --load-dir ../qwen-72b-hf \
                                    --save-dir {your megatron ckpt save path} \
                                    --tokenizer-model ../qwen-72b-hf/qwen.tiktoken \
                                    --add-qkv-bias
    
    cd ..
    ```

4. Prepare dataset

	Download the Qwen-72B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)    
	```shell
    # download datasets
    mkdir dataset_qwen-72b
    cd ./dataset_qwen-72b
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
	
	# process datasets                              
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

5. fine-tuning

	Config Qwen-72B fine-tuning script: examples/qwen/pretrain_qwen_72b_ptd.sh 
   ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    CKPT_SAVE_DIR="your model ckpt save path"
    TOKENIZER_MODEL="./qwen-72b-hf/"  #tokenizer path
    DATA_PATH="./dataset_qwen-72b/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="your megatron ckpt save path"
   ```

	Launch Qwen-72B fine-tuning script: examples/qwen/pretrain_qwen_72b_ptd.sh
   
   ```shell
    bash examples/qwen/pretrain_qwen_72b_ptd.sh 
   ```

### Performance

#### Machine performance

The performance of Qwen-72B in **Ascend NPU** and **Reference**:

|  Device   |  Model  | throughput rate (tokens/s/p)(8k) | throughput rate (tokens/s/p)(32k) |
|:---------:|:-------:|:--------------------------------:|:---------------------------------:|
|   NPUs    | Qwen-7B |               285                |                --                 |
| Reference | Qwen-7B |               345                |                --                 |
