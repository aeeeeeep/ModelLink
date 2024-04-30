# LLaMA
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [LLaMA](#llama)
- [Contents](#contents)
- [LLAMA3-8B](#llama3-8b)
  - [Training](#training)
    - [Script](#script)
    - [Performance](#performance)
      - [Machine performance](#machine-performance)
  - [Inference-8B](#inference-8b)
  - [Evaluation-8B](#evaluation-8b)

  - [LLaMA](#llama)
- [Contents](#contents)
- [LLAMA3-70B](#llama3-70b)
  - [Training](#training)
    - [Script](#script)
    - [Performance](#performance)
      - [Machine performance](#machine-performance)
  - [Inference-70B](#inference-70b)
  - [Evaluation-70B](#evaluation-70b)

# LLAMA3-8B

## Training

Here's a hardware summary of pre-training  LLAMA3-8B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout -f bcce6f
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

    # install AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```

    *Note that if you want to train with the weight from huggingface, please run fix a deepspeed loading checkpointing bug by modified `if zero_sd_list is None` as `if zero_sd_list is None or len(zero_sd_list) == 0` in the `_load_zero_checkpoint` function of `<deepspeed-installed-path>/runtime/engine.py`*

    ```text
    # original deepspeed/runtime/engine.py, about #Lines2746-2748
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None:
        return False

    # modified
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None or len(zero_sd_list) == 0:
        return False
    ```
3. Prepare pretrained weights and tokenizer
    Download the LLAMA3-8B checkpoint from [here](https://huggingface.co/unsloth/llama-3-8B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/llama-3-8b-hf/
    cd ./model_from_hf/llama-3-8b-hf/
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/generation_config.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/model-00001-of-00004.safetensors
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/model-00002-of-00004.safetensors
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/model-00003-of-00004.safetensors
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/model-00004-of-00004.safetensors
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/model.safetensors.index.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/special_tokens_map.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/tokenizer.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/tokenizer_config.json
    cd ../../
    ```
4. weight conversion in ptd mode

    *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-3-8b model weight conversion in ptd as an example.*

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # convert to ptd weights
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/llama-3-8b-hf/ \
        --save-dir ./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/llama-3-8b-hf/tokenizer.json
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-3-8b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-3-8b-hf/mg2hg/
    ```

    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.
5. pre-training

    5.1 Prepare dataset

    Download the LLAMA3-8B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./dataset/llama-3-8b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-8b-hf/ \
        --output-prefix ./dataset/llama-3-8b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 pre-training using ptd mode
    Config LLAMA3-8B pre-training script: examples/llama3/pretrain_llama3_8b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/llama-3-8b-hf/"
    TOKENIZER_MODEL="./model_from_hf/llama-3-8b-hf/"  #tokenizer path
    DATA_PATH="./dataset/llama-3-8b-hf/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/" #weight path
    ```

    Multi-machine training requires the addition of parameter --overlap-grad-reduce

    Launch LLAMA3-8B  pre-training script: examples/llama3/pretrain_llama3_8b_ptd.sh

    ```shell
    bash examples/llama3/pretrain_llama3_8b_ptd.sh 
    ```
    **Note**: If using multi machine training, it is necessary to set up multi machine data sharing, and non primary nodes can read the primary node data through data sharing. Alternatively, directly copy the data generated by the master node to non master nodes.

### Performance

#### Machine performance

The performance of LLaMA3-8B in **Ascend NPU** and **Reference**:

| Device      | Model         | total Iterations     | throughput rate (tokens/s/p) | 
| :------:    | :-----------: |:-------------------: | :-------------------------:  | 
| NPUs        | LLaMA3-8B     | 1000                 | 2483                         |
| Reference   | LLaMA3-8B     | 1000                 | 2674                         |



## Inference-8B

Config llama3-8B inference script: examples/llama3/generate_llama3_8b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"
```

Launch llama3-8B inference script: examples/llama3/generate_llama3_8b_ptd.sh

```bash
bash examples/llama3/generate_llama3_8b_ptd.sh
```


## Evaluation-8B

We use MMLU benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/cais/mmlu).
Config llama3-8B evaluation script: examples/llama3/evaluate_llama3_8b_ptd.sh

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script model path and tokenizer path
TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"  #tokenizer path
CHECKPOINT="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1"  #model path
# configure task and data path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Launch llama3-8B evaluation script:

```bash
bash examples/llama3/evaluate_llama3_8b_ptd.sh
```

Evaluation results

|  dataset | subject_num | question_num | reference_acc |NPU acc|
|:---:|:-----------:|:------------:|:-------------:|:---:|
| MMLU |     57      |    14042     |    0.666     |0.653|

# LLAMA3-70B

## Training

Here's a hardware summary of pre-training  LLAMA3-70B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               64 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout -f bcce6f
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

    # install AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```

    *Note that if you want to train with the weight from huggingface, please run fix a deepspeed loading checkpointing bug by modified `if zero_sd_list is None` as `if zero_sd_list is None or len(zero_sd_list) == 0` in the `_load_zero_checkpoint` function of `<deepspeed-installed-path>/runtime/engine.py`*

    ```text
    # original deepspeed/runtime/engine.py, about #Lines2746-2748
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None:
        return False

    # modified
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None or len(zero_sd_list) == 0:
        return False
    ```
3. Prepare pretrained weights and tokenizer
    Download the LLAMA3-70B checkpoint from [here](https://huggingface.co/v2ray/Llama-3-70B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/llama-3-70b-hf/
    cd ./model_from_hf/llama-3-70b-hf/
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/config.json
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/generation_config.json
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00001-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00002-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00003-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00004-of-00030.safetensors
    ...
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model-00030-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/model.safetensors.index.json
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/special_tokens_map.json
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/tokenizer.json
    wget https://huggingface.co/v2ray/Llama-3-70B/blob/main/tokenizer_config.json
    cd ../../
    ```
4. weight conversion in ptd mode

    *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-3-70b model weight conversion in ptd as an example.*

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # convert to ptd weights
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 8 \
        --load-dir ./model_from_hf/llama-3-70b-hf/ \
        --save-dir ./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/ \
        --tokenizer-model ./model_from_hf/llama-3-70b-hf/tokenizer.json
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-3-70b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-3-70b-hf/mg2hg/
    ```

    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.
5. pre-training

    5.1 Prepare dataset

    Download the LLAMA3-70B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./dataset/llama-3-70b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-70b-hf/ \
        --output-prefix ./dataset/llama-3-70b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 pre-training using ptd mode
    Config LLAMA3-70B pre-training script: examples/llama3/pretrain_llama3_70b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/llama-3-70b-hf/"
    TOKENIZER_MODEL="./model_from_hf/llama-3-70b-hf/"  #tokenizer path
    DATA_PATH="./dataset/llama-3-70b-hf/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/" #weight path
    ```

    Multi-machine training requires the addition of parameter --overlap-grad-reduce

    Launch LLAMA3-70B  pre-training script: examples/llama3/pretrain_llama3_70b_ptd.sh

    ```shell
    bash examples/llama3/pretrain_llama3_70b_ptd.sh 
    ```
    **Note**: If using multi machine training, it is necessary to set up multi machine data sharing, and non primary nodes can read the primary node data through data sharing. Alternatively, directly copy the data generated by the master node to non master nodes.

### Performance

#### Machine performance

The performance of LLaMA3-70B in **Ascend NPU** and **Reference**:

| Device      | Model         | total Iterations     | throughput rate (tokens/s/p) | 
| :------:    | :-----------: |:-------------------: | :-------------------------:  | 
| NPUs        | LLaMA3-70B     | 1000                 | 283                         |
| Reference   | LLaMA3-70B     | -                    | -                           |



## Inference-70B

Config llama3-70B inference script: examples/llama3/generate_llama3_70b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/llama-3-70b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"
```

Launch llama3-70B inference script: examples/llama3/generate_llama3_70b_ptd.sh

```bash
bash examples/llama3/generate_llama3_70b_ptd.sh
```


## Evaluation-70B

We use MMLU benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/cais/mmlu).
Config llama3-70B evaluation script: examples/llama3/evaluate_llama3_70b_ptd.sh

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script model path and tokenizer path
TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"  #tokenizer path
CHECKPOINT="./model_weights/llama-3-70b-hf-v0.1-tp8-pp1"  #model path
# configure task and data path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Launch llama3-70B evaluation script:

```bash
bash examples/llama3/evaluate_llama3_70b_ptd.sh
```

Evaluation results

|  dataset | subject_num | question_num | reference_acc |NPU acc|
|:---:|:-----------:|:------------:|:-------------:|:-------:|
| MMLU |     57      |    14042     |    0.795     |  0.783  |
