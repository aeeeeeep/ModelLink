# BaiChuan
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
    </p>
</p>


#  Contents

- [Baichuan-7B](#contents)
  - [Training](#pre-training)
    - [Script](#script)
    - [Performance](#performance)
        - [Machine performance](#machine-performance)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

- [Baichuan-13B](#contents)
  - [Training](#pre-training)
    - [Script](#script)
    - [Performance](#performance)
        - [Machine performance](#machine-performance)
  - [Lora](#Lora)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)


# Baichuan-7B

## Training

Here's a hardware summary of pre-training Baichuan-7B:

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
Download the Baichuan-7B checkpoint from [here](https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main) 

```shell
cd ./model_from_hf/
# 需要安装 git-lfs: git lfs install
git clone https://huggingface.co/baichuan-inc/Baichuan-7B
cd ..
```

4. Weights convert

In order to adapt to the baichuan-7B model, the following script is used to convert the model pre-training weights.
***(This scenario is generally used to train open-source HuggingFace models on Megatron)***
```shell
mkdir baichuan-7B-mt

# modify the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
python tools/checkpoint/util.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/Baichuan-7B/ \
    --save-dir ./model_weights/Baichuan-7B-v0.1-tp8-pp1/ \
    --tokenizer-model ./model_from_hf/Baichuan-7B/tokenizer.model \
    --w-pack True    
```
Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***
```shell
# Modify the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/Baichuan-7B-v0.1-tp8-pp1/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --w-pack True \
    --save-dir ./model_from_hf/Baichuan-7B/    # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Baichuan-7B/mg2hg
```

5. Prepare dataset

Download the Baichuan-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
# download datasets
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

# process datasets                              
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Baichuan-7B/ \
    --output-prefix ./dataset/Baichuan-7B_alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```


6. Config Baichuan-7B pre-training script : examples/baichuan/pretrain_baichuan_ptd_7B.sh 

```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

CKPT_SAVE_DIR="./ckpt/"
DATA_PATH="./dataset/baichuan-7B_alpaca_text_document"
TOKENIZER_MODEL="./model_from_hf/Baichuan-7B/tokenizer.model"
CKPT_LOAD_DIR="./model_weights/Baichuan-7B-v0.1-tp8-pp1/"
```


7. Launch Baichuan-7B  pre-training script: examples/baichuan/pretrain_baichuan_ptd_7B.sh 

```shell
bash examples/baichuan/pretrain_baichuan_ptd_7B.sh 
```



### Performance

#### Machine performance

The performance of Baichuan-7B in **Ascend NPU** and **Reference**:

| Device | Model       | total Iterations | throughput rate (samples/s) | throughput rate (tokens/s/p) | single-step time (s/step) | 
|:----:|:---------:|:----:|:---------------------:|:---------------:|:----------------:|
| NPUs | Baichuan-7B | 1000 |  5.24 | 2685 | 6.1| 
|  Reference  | Baichuan-7B | - | - |  2036 | - | 


## Inference
Config baichuan-7B inference script: tasks/inference/generate_baichuan_7b_ptd.sh
```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Baichuan-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Baichuan-7B/"
```
Launch baichuan-7B inference script: tasks/inference/generate_baichuan_7b_ptd.sh
```bash
bash tasks/inference/generate_baichuan_7b_ptd.sh
```
Some inference samples are as follows:

![Inference](../../sources/images/baichuan/baichuan_7B_inference.png)



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
bash ./tasks/evaluation/evaluate_baichuan_7B_ptd.sh
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
      <th>Baichuan 7B</th>
      <td>0.69</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/BoolQ">0.67</a></td>
    </tr>
  </tbody>
</table>


# Baichuan-13B

## Training

Here's a hardware summary of pre-training Baichuan-13B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs               |



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
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# modify the path according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

#install Ascendspeed
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
pip install -r requirements.txt 
pip3 install -e .
cd ..

# install other packages
pip install -r requirements.txt 
```

**Note:** If the error message "'AttributeError: 'BaichuanTokenizer' object has no attribute'sp_model'" is displayed during the script execution, run the following command to rectify the error:

```shell
pip install transformers==4.32.0 --force
```

3. Prepare pretrained weights


Download the Baichuan-13B checkpoint from [here](https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main) 

```shell
cd ./model_from_hf/
# 需要安装 git-lfs: git lfs install
git clone https://huggingface.co/baichuan-inc/Baichuan-13B-Base
cd ..
```

4. Weights convert

In order to adapt to the baichuan-13B model, the following script is used to convert the model pre-training weights.

***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

```shell

# modify the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
python tools/checkpoint/util.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --load-dir ./baichuan-13B-hf \
    --save-dir ./baichuan-13B-mt \
    --tokenizer-model ./baichuan-13B-hf/tokenizer.model \
    --w-pack True  
```

Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***
```shell
# Modify the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/Baichuan-13B-Base-v0.1-tp8-pp1/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --w-pack True \
    --save-dir ./model_from_hf/Baichuan-13B/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Baichuan-13B/mg2hg
```

5. Prepare dataset
Download the Baichuan-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

python ./tools/preprocess_data.py \
    --input ./dataset-baichuan-13B/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Baichuan-13B/ \
    --output-prefix ./dataset/Baichuan-13B_alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF 
```


6. Config Baichuan-13B pre-training script(Baichuan-13B does not support Flash Attention): examples/baichuan/pretrain_baichuan_ptd_13B.sh


```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

CKPT_SAVE_DIR="./ckpt/"
DATA_PATH="./dataset/Baichuan-13B_alpaca_text_document"
TOKENIZER_MODEL="./model_from_hf/Baichuan-13B/tokenizer.model"
CKPT_LOAD_DIR="./model_weights/Baichuan-13B-Base-v0.1-tp8-pp1/" 
```

7. Launch Baichuan-13B pre-training script: examples/baichuan/pretrain_baichuan_ptd_13B.sh

```bash
bash examples/baichuan/pretrain_baichuan_ptd_13B.sh
```


### Performance

#### Machine performance

The performance of the Baichuan-13B in **Ascend NPU** and **Reference**:

| Device |    Model     | total Iterations | throughput rate (samples/s) | throughput rate (tokens/s/p) | single-step time (s/step) | 
|:----:|:------------:|:----:|:------------------:|:--------------------:|:---------------:|
| NPUs | Baichuan-13B | 1000 | 2.37 | 1213| 13.5 |    
|  Reference | Baichuan-13B |  - |  -   | 862  |  -   |    



## Inference
Config baichuan-13B inference script: tasks/inference/generate_baichuan_13b_ptd.sh
```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Baichuan-13B-Base-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Baichuan-13B/"
```

Launch baichuan-13B inference script: tasks/inference/generate_baichuan_13b_ptd.sh
```bash
bash tasks/inference/generate_baichuan_13b_ptd.sh
```
Some inference samples are as follows:
![Inference](../../sources/images/baichuan/baichuan_13B_inference.png)



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
bash ./tasks/evaluation/evaluate_baichuan_13B_ptd.sh
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
      <th>Baichuan 13B</th>
      <td>0.747</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/BoolQ">0.736</a></td>
    </tr>
  </tbody>
</table>
