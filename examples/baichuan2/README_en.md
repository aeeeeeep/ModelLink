# BaiChuan2
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
    </p>
</p>


#  Contents
- [Baichuan2-7B](#contents)
  - [Training](#pre-training)
    - [Script](#script)
    - [Performance](#performance)
        - [Machine performance](#machine-performance)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)
  
- [Baichuan2-13B](#contents)
  - [Training](#pre-training)
    - [Script](#script)
    - [Performance](#performance)
        - [Machine performance](#machine-performance)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

# Baichuan2-7B

## Training

Here's a hardware summary of pre-training Baichuan2-7B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |


### Script

1. Clone the repository to your local server:
```shell
git clone https://gitee.com/ascend/ModelLink.git 
cd ModeLlink 
mkdir logs
mkdir ckpt
```

2. Build environment

```bash
# python3.8
conda create -n test python=3.8
conda activate test

# install torch and torch_npu 
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# modify the path according to your own  ascend-toolkit path
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


3. Prepare pretrained weights
Download the Baichuan2-7B checkpoint from [here](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main)



```shell
mkdir baichuan2-7B-hf
cd ./baichuan2-7B-hf
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/generation_utils.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model-00001-of-00002.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model-00002-of-00002.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/quantizer.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenizer.model
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenizer_config.json
cd ..
```

4. Weights convert

In order to adapt to the baichuan2-7B model, the following script is used to convert the model pre-training weights.
***(This scenario is generally used to train open-source HuggingFace models on Megatron)***
```shell
mkdir baichuan2-7B-mt

# modify the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
python tools/checkpoint/util.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --load-dir ./baichuan2-7B-hf \
    --save-dir ./baichuan2-7B-mt \
    --tokenizer-model ./baichuan2-7B-hf/tokenizer.model \
    --w-pack True  
```
Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***
```shell
cd ModelLink/
# Modify the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ../Baichuan2-7B-v0.1-pt8-pp1 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --w-pack True \
    --save-dir ../Baichuan2-7B_downloaded   # <-- Fill in the original HF model path here, new weights will be saved in ../Baichuan2-7B_downloaded/mg2hg
```


5. Prepare dataset

Download the Baichuan2-7B-Base datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
# download datasets
mkdir dataset-baichuan2-7B
cd ./dataset-baichuan2-7B
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

# process datasets                              
python ./tools/preprocess_data.py \
--input ./dataset-baichuan2-7B/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
--tokenizer-name-or-path ./baichuan2-7B-hf \
--output-prefix ./dataset-baichuan2-7B/alpaca \
--workers 4 \
--log-interval 1000 \
--tokenizer-type PretrainedFromHF
```


6. Config Baichuan2-7B pre-training script : examples/baichuan2/pretrain_baichuan2_ptd_7B.sh 

```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script orign dataset path according to your own dataset path
CKPT_SAVE_DIR="./ckpt"
DATA_PATH="./dataset-baichuan2-7B/alpaca_text_document"
TOKENIZER_MODEL="./baichuan2-7B-hf/tokenizer.model"
CKPT_LOAD_DIR="./baichuan2-7B-mt"
```

 
7. Launch Baichuan2-7B  pre-training script: examples/baichuan2/pretrain_baichuan2_ptd_7B.sh 

```shell
bash examples/baichuan2/pretrain_baichuan2_ptd_7B.sh 
```


### Performance

#### Machine performance

The performance of Baichuan2-7B in **Ascend NPU** and **Reference**:

| Device | Model       | total Iterations | throughput rate (samples/s) | throughput rate (tokens/s/p) | single-step time (s/step) | 
|:----:|:---------:|:----:|:---------------------:|:---------------:|:----------------:|
| NPUs | Baichuan2-7B | 1000 | 5.2 | 2664 | 12.3| 
|  Reference  | Baichuan2-7B | 1000 | -- |  3969 | --   | 



## Inference
Config baichuan2-7B inference script: tasks/inference/generate_baichuan2_7b_ptd.sh
```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
```
Launch baichuan2-7B inference script: tasks/inference/generate_baichuan2_7b_ptd.sh
```bash
bash tasks/inference/generate_baichuan2_7b_ptd.sh
```
Some inference samples are as follows:
![Inference](../../sources/images/baichuan2/baichuan2_7B_inference.png)


## Evaluation
We use the boolq benchmark to evaluate our model. Benchmark [Download](https://huggingface.co/datasets/boolq).

```shell
# config origin weight and vocab file path
CHECKPOINT=<origin-ckpt-path>
TOKENIZER_PATH=<tokenizer-path>
# config tasks and dataset path
DATA_PATH="./boolq/"
TASK="boolq"
```

```shell
bash ./tasks/evaluation/evaluate_baichuan2_13B_ptd.sh
```

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Subset</th>
      <th>Model</th>
      <th>NPU</th>
      <th>OpenSource</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">Boolq</a></td>
      <td>test</td>
      <th>Baichuan2-7B</th>
      <td>0.7</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/BoolQ">0.632</a></td>
    </tr>
  </tbody>
</table>

# Baichuan2-13B

## Training

Here's a hardware summary of pre-training Baichuan2-13B:

| Hardware |      Value       |
| :------: |:----------------:|
|   NPU    | 8 x Ascend NPUs |



### Script
1. Clone the repository to your local server:
```shell
git clone https://gitee.com/ascend/ModelLink.git 
cd ModeLlink 
mkdir logs
mkdir ckpt
```

2. Build environment

```bash
# python3.8
conda create -n test python=3.8
conda activate test

# install torch and torch_npu 
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# modify the path according to your own  ascend-toolkit path
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

3. Prepare pretrained weights


Download the Baichuan2-13B checkpoint from [here](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/tree/main) 
```shell
mkdir baichuan2-13B-hf
cd ./baichuan2-13B-hf
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/generation_utils.py
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/pytorch_model-00001-of-00003.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/pytorch_model-00002-of-00003.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/pytorch_model-00003-of-00003.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/quantizer.py
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/tokenizer_config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/resolve/main/tokenizer.model
cd ..
```

4. Weights convert

In order to adapt to the baichuan2-13B model, the following script is used to convert the model pre-training weights.
***(This scenario is generally used to train open-source HuggingFace models on Megatron)***
```shell
mkdir baichuan2-13B-mt

# modify the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
python tools/checkpoint/util.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --load-dir ./baichuan2-13B-hf \
    --save-dir ./baichuan2-13B-mt \
    --tokenizer-model ./baichuan2-13B-hf/tokenizer.model \
    --w-pack True  
```

Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***
```shell
cd ModelLink/
# Modify the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ../Baichuan2-13B-v0.1-pt8-pp1 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --w-pack True \
    --save-dir ../Baichuan2-13B_downloaded   # <-- Fill in the original HF model path here, new weights will be saved in ../Baichuan2-13B_downloaded/mg2hg
```

5. Prepare dataset

Download the Baichuan2-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
mkdir dataset-baichuan2-13B
cd ./dataset-baichuan2-13B
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

python ./tools/preprocess_data.py \
    --input ./dataset-baichuan2-13B/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./baichuan2-13B-hf \
    --output-prefix ./dataset-baichuan2-13B/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF 
```

6. Config Baichuan2-13B pre-training script: examples/baichuan2/pretrain_baichuan2_ptd_13B.sh

```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script orign dataset path according to your own dataset path
CKPT_SAVE_DIR="./ckpt"
DATA_PATH="./dataset-baichuan2-13B/alpaca_text_document"
TOKENIZER_MODEL="./baichuan2-13b-hf/tokenizer.model"
CKPT_LOAD_DIR="./baichuan2-13b-mt" 
```
7. Launch Baichuan2-13B pre-training script: examples/baichuan2/pretrain_baichuan2_ptd_13B.sh

```bash
bash examples/baichuan2/pretrain_baichuan2_ptd_13B.sh
```


### Performance

#### Machine performance

The performance of the Baichuan2-13B in **Ascend NPU** and **Reference**:

| Device |     Model     | total Iterations | throughput rate (samples/s/p) | throughput rate (tokens/s/p) | single-step time (s/step) | 
|:----:|:-------------------------:|:----:|:------------------:|:----------------------------:|:---------------:|
| NPUs | Baichuan2-13B |1000 |1.83|             1346             | 4.35 |
|  Reference  | Baichuan2-13B |-|-|             872              |- |



## Inference
Config baichuan2-13B inference script: tasks/inference/generate_baichuan2_13b_ptd.sh
```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
```
Launch baichuan2-13B inference script: tasks/inference/generate_baichuan2_13b_ptd.sh
```bash
bash tasks/inference/generate_baichuan2_13b_ptd.sh
```
Some inference samples are as follows:
![Inference](../../sources/images/baichuan2/baichuan2_13B_inference.png)


## Evaluation
We use the boolq benchmark to evaluate our model. Benchmark [Download](https://huggingface.co/datasets/boolq).

```shell
# config origin weight and vocab file path
CHECKPOINT=<origin-ckpt-path>
TOKENIZER_PATH=<tokenizer-path>
# config tasks and dataset path
DATA_PATH="./boolq/"
TASK="boolq"
```

```shell
bash ./tasks/evaluation/evaluate_baichuan2_13B_ptd.sh
```

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Subset</th>
      <th>Model</th>
      <th>NPU</th>
      <th>OpenSource</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">Boolq</a></td>
      <td>test</td>
      <th>Baichuan2-13B</th>
      <td>0.78</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/BoolQ">0.67</a></td>
    </tr>
  </tbody>
</table>
