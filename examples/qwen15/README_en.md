# Qwen1.5 $\color{black}{\rm\tiny{【Model}}$ $\color{black}{\rm\tiny{contributed}}$ $\color{black}{\rm\tiny{by}}$ $\color{black}{\rm\tiny{community】}}$
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [Qwen1.5-7B](#qwen15-7b)
  - [Training](#training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

- [Qwen1.5-14B](#qwen15-14b)
  - [Training](#training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

# Qwen1.5-7B

## Training

Here's a hardware summary of pre-training  Qwen1.5-7B:

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
    pip install torch-2.2.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.2.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # install AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
    pip install -r requirements.txt
    pip install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt
    ```
    **Note** : transformer version 4.37.0 or higher
3. Prepare pretrained weights and tokenizer
   Download the Qwen1.5-7B checkpoint from [here](https://huggingface.co/Qwen/Qwen1.5-7B/tree/main)

   ```bash
   mkdir ./model_from_hf/Qwen1.5-7B/
   cd ./model_from_hf/Qwen1.5-7B/
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/merges.txt
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00001-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00002-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00003-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00004-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/special_tokens_map.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/tokenizer.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/tokenizer_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/vocab.json
   cd ../../
   ```

4. Weights convert

    Convert weights from huggingface format to megatron format
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --make-vocab-size-divisible-by 16 \
        --load-dir ./model_from_hf/Qwen1.5-7B/ \
        --save-dir ./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Qwen1.5-7B/tokenizer.json \
        --add-qkv-bias \
        --param-dtype bf16 
    ```

   Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_qwen \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --load-dir ./model_weights/Qwen1.5-7B-v0.1-tp8-pp1 \
        --save-dir ./model_from_hf/Qwen1.5-7B   # Fill in the original HF model path here, new weights will be saved in 1.5/mg2hg/
    ```
5. Pre-training

   5.1 prepare dataset

   Download the Qwen1.5-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # download datasets
   cd ./dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..

   # process datasets
   mkdir ./dataset/Qwen1.5-7B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-7B \
       --output-prefix ./dataset/Qwen1.5-7B/alpaca \
       --tokenizer-type PretrainedFromHF \
       --seq-length 8192 \
       --workers 4 \
       --log-interval 1000
   ```
   5.2 pre-training

   Config Qwen1.5-7B pre-training script: examples/qwen15/pretrain_qwen15_7b_ptd.sh

   ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/Qwen1.5-7B/"
    TOKENIZER_MODEL="./model_from_hf/Qwen1.5-7B"  #tokenizer path
    DATA_PATH="./dataset/Qwen1.5-7B/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1"
   ```
   Multi-machine training requires the addition of parameter `--overlap-grad-reduce`.

   Launch Qwen1.5-7B pre-training script: examples/qwen15/pretrain_qwen15_7b_ptd.sh

   ```shell
    bash examples/qwen15/pretrain_qwen15_7b_ptd.sh 
   ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

6. fine-tuning

   6.1 Prepare fine-tuning dataset Download the Qwen1.5-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # download datasets
   mkdir finetune_dataset
   cd ./finetune_dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..

   # process datasets   
   mkdir ./finetune_dataset/Qwen1.5-7B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-7B/ \
       --output-prefix ./finetune_dataset/Qwen1.5-7B/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF \
       --handler-name GeneralInstructionHandler \
       --append-eod
   ```
   
   6.2 Full Parameters Fine-Tuning

   The configuration script with the fine-tuning parameters is basically the same as the pre-training script.

   *The difference is the dataset, and add the training parameter `--is-instruction dataset`, add the fine-tuning parameter `--finetune`, add the pre-training weight loading parameter `--load`, so that the fine-tuning starts from the first step, modify the tokenizer parameter.*

    Modified as follows:

   ```bash
   CKPT_LOAD_DIR="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/"
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-7B/"
   DATA_PATH="./finetune_dataset/Qwen1.5-7B/alpaca"
   TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B/"
   
   --load ${CKPT_PATH} \
   --finetune \
   --is-instruction-dataset \
   --tokenizer-not-use-fast \
   ```
   
   Launch Qwen1.5-7B fine-tuning script: examples/qwen15/tune_qwen15_7b_ptd.sh

   ```shell
    bash examples/qwen15/tune_qwen15_7b_ptd.sh
   ```

### Performance

#### Machine performance

The performance of Qwen1.5-7B in **Ascend NPU** and **Reference**:

|     Device     |      Model       | throughput rate (tokens/s/p)  |
|:--------------:|:----------------:|:-----------------------------:|
|      NPUs      |    Qwen1.5-7B    |             2862              |
|   Reference    |    Qwen1.5-7B    |             2621              |

## Inference

Config Qwen1.5-7B inference script: examples/qwen15/generate_qwen15_7b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B"
```

Launch Qwen1.5-7B inference script: examples/qwen15/generate_qwen15_7b_ptd.sh

```bash
bash examples/qwen15/generate_qwen15_7b_ptd.sh
```

**Note**: If using multi machine training, it is necessary to set up multi machine data sharing, and non primary nodes can read the primary node data through data sharing. Alternatively, directly copy the data generated by the master node to non master nodes.

Some inference samples are as follows:
![Inference](../../sources/images/qwen15/qwen1.5_7b_inference.png)

## Evaluation

We use the [CEval benchmark](https://huggingface.co/datasets/ceval/ceval-exam) and [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) to evaluate our model.

Config Qwen1.5-7B evaluation script: examples/qwen15/evaluate_qwen15_7b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify the model parameter path and vocabulary path
TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B/"  # vocabulary path
CHECKPOINT="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/"  # parameter path

# Configure the task type and dataset path
DATA_PATH="./mmlu/data/test/"  # "./ceval/val/" for ceval task
TASK="mmlu"  # "ceval" for ceval task
```

Launch Qwen1.5-7B evaluation

```bash
bash examples/qwen15/evaluate_qwen15_7b_ptd.sh
```

| Task  | Subset | Question |                          OpenSource                           |   NPU   |
|:-----:|:------:|:--------:|:-------------------------------------------------------------:|:-------:|
| MMLU  |   57   |  14042   |       [61.0](https://qwenlm.github.io/zh/blog/qwen1.5)        |  60.3   |


# Qwen1.5-14B

## Training

Here's a hardware summary of pre-training  Qwen1.5-14B:

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
    pip install torch-2.2.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.2.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # install AscendSpeed
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
    pip install -r requirements.txt
    pip install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt
    ```
    **Note** : transformer version 4.37.0 or higher
3. Prepare pretrained weights and tokenizer
   Download the Qwen1.5-14B checkpoint from [here](https://huggingface.co/Qwen/Qwen1.5-14B/tree/main)

   ```bash
   mkdir ./model_from_hf/Qwen1.5-14B/
   cd ./model_from_hf/Qwen1.5-14B/
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/config.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/merges.txt
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/special_tokens_map.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/tokenizer.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/tokenizer_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/vocab.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/model-00001-of-00008.safetensors
   ...
   cd ../../
   ```

4. Weights convert

    Convert weights from huggingface format to megatron format
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --make-vocab-size-divisible-by 16 \
        --load-dir ./model_from_hf/Qwen1.5-14B/ \
        --save-dir ./model_weights/Qwen1.5-14B-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Qwen1.5-14B/tokenizer.json \
        --add-qkv-bias \
        --param-dtype bf16 
    ```

   Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_qwen \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --load-dir ./model_weights/Qwen1.5-14B-v0.1-tp8-pp1 \
        --save-dir ./model_from_hf/Qwen1.5-14B   # Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Qwen1.5-14B/mg2hg/
    ```
5. Pre-training

   5.1 prepare dataset

   Download the Qwen1.5-14B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # download datasets
   cd ./dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..

   # process datasets
   mkdir ./dataset/Qwen1.5-14B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-14B \
       --output-prefix ./dataset/Qwen1.5-14B/alpaca \
       --tokenizer-type PretrainedFromHF \
       --seq-length 8192 \
       --workers 4 \
       --log-interval 1000
   ```
   5.2 pre-training

   Config Qwen1.5-14B pre-training script: examples/qwen15/pretrain_qwen15_14b_ptd.sh

   ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/Qwen1.5-14B/"
    TOKENIZER_MODEL="./model_from_hf/Qwen1.5-14B"  #tokenizer path
    DATA_PATH="./dataset/Qwen1.5-14B/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/Qwen1.5-14B-v0.1-tp8-pp1"
   ```
   Multi-machine training requires the addition of parameter `--overlap-grad-reduce`.

   Launch Qwen1.5-14B pre-training script: examples/qwen15/pretrain_qwen15_14b_ptd.sh

   ```shell
    bash examples/qwen15/pretrain_qwen15_14b_ptd.sh 
   ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

6. fine-tuning

   6.1 Prepare fine-tuning dataset Download the Qwen1.5-14B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # download datasets
   mkdir finetune_dataset
   cd ./finetune_dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..

   # process datasets   
   mkdir ./finetune_dataset/Qwen1.5-14B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-14B/ \
       --output-prefix ./finetune_dataset/Qwen1.5-14B/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF \
       --handler-name GeneralInstructionHandler \
       --append-eod
   ```
   
   6.2 Full Parameters Fine-Tuning

   The configuration script with the fine-tuning parameters is basically the same as the pre-training script.

   *The difference is the dataset, and add the training parameter `--is-instruction dataset`, add the fine-tuning parameter `--finetune`, add the pre-training weight loading parameter `--load`, so that the fine-tuning starts from the first step, modify the tokenizer parameter.*

    Modified as follows:

   ```bash
   CKPT_LOAD_DIR="./model_weights/Qwen1.5-14B-v0.1-tp8-pp1/"
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-14B/"
   DATA_PATH="./finetune_dataset/Qwen1.5-14B/alpaca"
   TOKENIZER_PATH="./model_from_hf/Qwen1.5-14B/"
   
   --load ${CKPT_PATH} \
   --finetune \
   --is-instruction-dataset \
   --tokenizer-type PretrainedFromHF \
   --tokenizer-name-or-path ${TOKENIZER_PATH} \
   --tokenizer-not-use-fast \
   ```
   
   Launch Qwen1.5-14B fine-tuning script: examples/qwen15/tune_qwen15_14b_ptd.sh

   ```shell
    bash examples/qwen15/tune_qwen15_14b_ptd.sh
   ```

### Performance

#### Machine performance

The performance of Qwen1.5-14B in **Ascend NPU** and **Reference**:

|  Device   |    Model    | throughput rate (tokens/s/p) |
|:---------:|:-----------:|:----------------------------:|
|   NPUs    | Qwen1.5-14B |            1717.8            |
| Reference | Qwen1.5-14B |            1702.2            |

## Inference

Config Qwen1.5-14B inference script: examples/qwen15/generate_qwen15_14b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Qwen1.5-14B-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/Qwen1.5-14B"
```

Launch Qwen1.5-14B inference script: examples/qwen15/generate_qwen15_14b_ptd.sh

```bash
bash examples/qwen15/generate_qwen15_14b_ptd.sh
```

**Note**: If using multi machine training, it is necessary to set up multi machine data sharing, and non primary nodes can read the primary node data through data sharing. Alternatively, directly copy the data generated by the master node to non master nodes.

Some inference samples are as follows:
![Inference](../../sources/images/qwen15/qwen1.5_14b_inference.png)

## Evaluation

We use the [CEval benchmark](https://huggingface.co/datasets/ceval/ceval-exam) and [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) to evaluate our model.

Config Qwen1.5-14B evaluation script: examples/qwen15/evaluate_qwen15_14b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify the model parameter path and vocabulary path
TOKENIZER_PATH="./model_from_hf/Qwen1.5-14B/"  # vocabulary path
CHECKPOINT="./model_weights/Qwen1.5-14B-v0.1-tp8-pp1/"  # parameter path

# Configure the task type and dataset path
DATA_PATH="./mmlu/data/test/"  # "./ceval/val/" for ceval task
TASK="mmlu"  # "ceval" for ceval task
```

Launch Qwen1.5-14B evaluation

```bash
bash examples/qwen15/evaluate_qwen15_14b_ptd.sh
```

| Task  | Subset | Question |                    OpenSource                    | NPU  |
|:-----:|:------:|:--------:|:------------------------------------------------:|:----:|
| MMLU  |   57   |  14042   | [67.6](https://qwenlm.github.io/zh/blog/qwen1.5) | 67.3 |

