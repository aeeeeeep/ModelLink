# LLaMA-7B/13B

<p align="left">
        <b><a href="https://gitee.com/ascend/ModelLink/blob/modellink/examples/llama/README.md">简体中文</a></b> |
        <b>English</b>
</p>


- [LLaMA-7B/13B](#llama-7b13b)
  - [Training](#training)
    - [Script](#script)
    - [Performance](#performance)
      - [Machine performance](#machine-performance)
      - [Accuracy of the loss](#accuracy-of-the-loss)
  - [Inference](#Inference)
    - [Script](#script)
  - [Evaluation with Numerous Benchmarks](#Evaluation-with-Numerous-Benchmarks)
- [LLaMA-33B/65B](#llama-65b)
  - [Training](#pre-training)
    - [Datasets](#datasets)
    - [Script](#script-1)
    - [Performance](#performance-1)
      - [Machine performance](#machine-performance-1)
      - [Accuracy of the loss](#accuracy-of-the-loss-1)
  - [Inference](#Inference)
    - [Script](#script)
  - [Evaluation with Numerous Benchmarks](#Evaluation-with-Numerous-Benchmarks)
- [Citation](#citation)

## Training

Here's a hardware summary of pre-training LLaMA-7B/13B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |


### Script

1. Clone the repository to your local server:
```shell
git clone -b modellink https://gitee.com/ascend/ModelLink.git 
cd ModelLink
mkdir logs
cd ..
```

2. Build environment

```bash
# python3.8
conda create -n test python=3.8
conda activate test
# install torch and torch_npu
pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
# modify the path according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# install ascendspeed
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
pip install -r requirements.txt
pip3 install -e .
cd ..

# install other packages
pip install -r requirements.txt 
```

3. Download the LLaMA-7B/13B tokenizer model file and checkpoint.

Download the LLaMA-7B checkpoint from [here](https://huggingface.co/ruibin-wang/llama-7b-hf/tree/main) 
```shell
  mkdir model_from_hf
  cd ./model_from_hf
  # you must install git-lfs
  git clone https://huggingface.co/ruibin-wang/llama-7b-hf
  cd ..
```

Download the LLaMA-13B checkpoint from [here](https://huggingface.co/ruibin-wang/llama-13b-hf/tree/main) 
```shell
  mkdir model_from_hf
  cd ./model_from_hf
  # you must install git-lfs
  git clone https://huggingface.co/ruibin-wang/llama-13b-hf
  cd ..
```

In order to adapt to the LLaMA-7B/13B model, the following script is used to convert the model pre-training weights.

LLaMA-7B
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir model_weights
python tools/checkpoint/util.py --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 8 \
    --load-dir ./model_from_hf/llama-7b-hf/ \
    --save-dir ./model_weights/llama-7b-tp1-pp8/ \
    --tokenizer-model ./model_from_hf/llama-7b-hf/tokenizer.model
```

LLaMA-13B
```shell
# Single machine with 8p
mkdir model_weights
python tools/checkpoint/util.py --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 8 \
    --load-dir ./model_from_hf/llama-13b-hf/ \
    --save-dir ./model_weights/llama-13b-tp1-pp8/ \
    --tokenizer-model ./model_from_hf/llama-13b-hf/tokenizer.model
```

4. Pretrain

4.1 Prepare pretrain dataset
Download the LLaMA-7B/13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

Preprocess dataset

LLaMA-7B
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-7b-hf \
    --output-prefix ./dataset/llama \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF  
```

LLaMA-13B
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-13b-hf \
    --output-prefix ./dataset/llama \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF  
```

4.2 Config LLaMA-7B/13B pre-training script.

LLaMA-7B
```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# modify script orign dataset path according to your own dataset path
TOKENIZER_MODEL=./model_from_hf/llama-7b-hf/tokenizer.model 
DATA_PATH=./dataset/llama_text_document  #processed dataset
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

LLaMA-13B
```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# modify script orign dataset path according to your own dataset path
TOKENIZER_MODEL=./model_from_hf/llama-13b-hf/tokenizer.model 
DATA_PATH=./dataset/llama_text_document  #processed dataset
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

4.3 Launch LLaMA-7B/13B pre-training script.

LLaMA-7B
```shell
bash examples/llama/pretrain_llama_7b_ptd.sh
```

LLaMA-13B
```shell
# 8p
bash examples/llama/pretrain_llama_13b_ptd.sh 
```


5. Pretrain

5.1 Prepare fine-tune dataset
Download the LLaMA-7B/13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

Preprocess instruction dataset

LLaMA-7B
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/llama-7b-hf \
  --output-prefix ./finetune_dataset/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralInstructionHandler \
  --append-eod
```

LLaMA-13B
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/llama-13b-hf \ 
  --output-prefix ./finetune_dataset/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralInstructionHandler \
  --append-eod
```

5.2 Config LLaMA-7B/13B fine-tune script.

LLaMA-7B
```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# modify script orign dataset path according to your own dataset path
TOKENIZER_MODEL=./model_from_hf/llama-7b-hf/  #tokenizer path
DATA_PATH=./finetune_dataset/alpaca  #processed dataset
LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

LLaMA-13B
```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# modify script orign dataset path according to your own dataset path
TOKENIZER_MODEL=./model_from_hf/llama-13b-hf/  #tokenizer path
DATA_PATH=./finetune_dataset/alpaca  #processed dataset
LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

5.3 Launch LLaMA-7B/13B fine-tune script.

LLaMA-7B
```shell
bash examples/llama/tune_llama_7b_ptd.sh
```

LLaMA-13B
```shell
# 8p
bash examples/llama/tune_llama_13b_ptd.sh 
```


### Performance

#### Machine performance

The performance of LLaMA-7B/13B in **Ascend NPU** and **Reference**:

| Device    | Hardware  | Model     | total Iterations | throughput rate (samples/s/p) | throughput rate (tokens/s/p) | single-step time (s/step) | floating point operation (TFLOPs/s) |
|-----------|-----------|-----------|------------------|-------------------------------|------------------------------|---------------------------|-------------------------------------|
| NPUs      | 910 1*8p  | LLaMA-7B  | 2048             | 1.75                          | 3600                         | 18.2                      | 159.9                               |
| Reference | -         | LLaMA-7B  | 2048             | 1.85                          | 3804                         | 18.5                      | 161.5                               |
| NPUs      | 910 1*8p  | LLaMA-13B | 2048             | 0.92                         | 1895                         | 17.2                     | 200.57                              |
| Reference | -         | LLaMA-13B | 2048             | 0.96                          | 2012                         | 16.6                     | 213.29                              |



#### Accuracy of the loss

LLama-7b with huggingface weights NPU vs GPU loss.
![NPU-Loss-with-weight-and-Relative-Error](../../sources/images/llama/llama7b-loss-with-weight.png)

LLama-13b with huggingface weights NPU vs GPU loss.
![NPU-Loss-with-weight-and-Relative-Error](../../sources/images/llama/llama13b-loss-with-weight.png)


## Inference

We support ModelLink Inference for text generation with LLaMA-7B and LLaMA-13B.
Inference different from pre-training, such as we need to Load pre-training checkpoint and the length of the output samples:

Config LLaMA-7B inference script `tasks/inference/generate_llama_7B_ptd.sh` and LLaMA-13B inference script `tasks/inference/generate_llama_13B_ptd.sh`.

```shell
# modify the model weight path and tokenizer path
CHECKPOINT=<checkpoint-path>
VOCAB_FILE=<vocabfile-path>
```

LLaMA-7B:
```shell
bash ./tasks/inference/generate_llama_7B_ptd.sh
```

LLaMA-13B:
```shell
bash ./tasks/inference/generate_llama_13B_ptd.sh
```

Some inference samples are as follows:

LLaMA-7B:

![llama-7B_generate.png](../../sources/images/llama/llama-7B_generate.png)

LLaMA-13B:

![llama-13B_generate.png](../../sources/images/llama/llama-13B_generate.png)


## Evaluation with Numerous Benchmarks

We use boolq benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/boolq).

Config LLaMA-7B evaluation script `tasks/evaluation/evaluate_llama_7B_ptd.sh` and LLaMA-13B evaluation script `tasks/evaluation/evaluate_llama_13B_ptd.sh`:

Modify checkpoint path, vocab path, dataset path and task:

```shell
CHECKPOINT=<checkpoint-path>
VOCAB_FILE=<vocabfile-path>
DATA_PATH="./boolq/data/test/"
TASK="boolq"
```
Change the max new tokens:
```shell
--max-new-tokens 1 
```

```text
# Note that, a deepspeed bug needs to be fixed during evaluation：
# Comment out line 671 in the file `<deepspeed-installed-path>/runtime/pipe/engine.py`：
# self.total_loss += self.loss.detach()
```

Start evaluation:
```shell
bash tasks/evaluation/evaluate_llama_7B_ptd.sh
bash tasks/evaluation/evaluate_llama_13B_ptd.sh
```

The evaluation performance of LLaMA-7B/13B in **Ascend NPU**:

| Task    | Model     | NPU  | Benchmark                                               |
|---------|-----------|------|---------------------------------------------------------|
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-7B  | 74.6 | [75.4](https://hub.opencompass.org.cn/dataset-detail/BoolQ) | 
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-13B | 79.6 | [78.7](https://hub.opencompass.org.cn/dataset-detail/BoolQ)   |

# LLaMA-33B/65B

This directory contains some of the scripts that were used to produce the results in the ModelLink. These scripts is to show the example how to run llama-65B in terminal.

LLaMA model is from: [LLaMA: OPen and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971v1.pdf)

>Touvron, Hugo, et al. "LLaMA: OPen and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971 (2023).

## Training

LLaMA's model performace is better than GPT3 with less parameters. The 33B/65B LLaMA model is comparable to Google's Chinchilla-70B and Palm-540B.

Here's a hardware summary of training llama:

| Hardware |      Value       |
| :------: |:----------------:|
|   NPU    | 32 x Ascend NPUs |


### Datasets
The model was trained using alpaca datasets.

### Script
1. Clone the repository to your local server
```shell
git clone -b modellink https://gitee.com/ascend/ModelLink.git 
cd ModelLink
mkdir logs
mkdir ckpt
```

2. Install ModelLink requirement environment.
```bash
# python3.8
conda create -n test python=3.8
conda activate test
# install torch and torch_npu
wget https://download.pytorch.org/whl/torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
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

3. Download tokenizer model file and checkpoint

llama-33B checkpoint
```shell
mkdir tokenizer
cd ./tokenizer

# make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/pinkmanlove/llama-33b-hf
cd ..
# revise "LLaMATokenizer" as "LLaMTokenizer" in tokenizer_config.json
```

llama-65B checkpoint
```shell
mkdir model_from_hf
cd ./model_from_hf

# make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/pinkmanlove/llama-65b-hf
cd ..
# revise "LLaMATokenizer" as "LLaMTokenizer" in tokenizer_config.json
```
In order to adapt to llama-33B/65B model, the following script is used to convert the model pre-training weights

llama-33B
```shell
mkdir model_weights

python tools/checkpoint/util.py --model-type GPT \
                                --loader llama2_hf \
                                --saver megatron \
                                --target-tensor-parallel-size 4 \
                                --target-pipeline-parallel-size 4 \
                                --load-dir ./model_from_hf/llama-33b-hf \
                                --save-dir ./model_weights/llama-33b \
                                --tokenizer-model ./model_from_hf/llama-33b-hf/tokenizer.model
```

llama-65B
```shell
mkdir model_weights
python tools/checkpoint/util.py --model-type GPT \
                                --loader llama2_hf \
                                --saver megatron \
                                --target-tensor-parallel-size 8 \
                                --target-pipeline-parallel-size 4 \
                                --load-dir ./model_from_hf/llama-65b-hf \
                                --save-dir ./model_weights/llama-65b-tp8-pp4 \
                                --tokenizer-model ./model_from_hf/llama-65b-hf/tokenizer.model
```

4. Pretrain

4.1 Prepare pretrain dataset
```shell
# for llama, dowload alpaca dataset, like
wget http://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
```

LLaMA-33B
```shell
mkdir dataset
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-33b-hf \
    --output-prefix ./dataset/llama \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF 
```

LLaMA-65B
```shell
mkdir dataset
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-65b-hf \
    --output-prefix ./dataset/llama \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF 
```

4.2 Config llama-33B/65B pre-training script :

Config llama-33B pre-training script `./examples/llama/pretrain_llama_33B_ptd_32p.sh`:
```bash
# modify the script according to your own conda and ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# modify script orign dataset path according to your own dataset path
TOKENIZER_MODEL=./tokenizer/llama-33b-hf/tokenizer.model
DATA_PATH=./dataset/llama_text_document 
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

Config llama-65B pre-training script `./examples/llama/pretrain_llama_65b_ptd.sh`:
```bash
# modify the script according to your own conda and ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# modify script orign dataset path according to your own dataset path
TOKENIZER_MODEL=./model_from_hf/llama-65b-hf/tokenizer.model
DATA_PATH=./dataset/llama_text_document 
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

4.3 Launch  pre-training script:

Launch llama-33B pre-training script : ModelLink/examples/llama/pretrain_llama_33B_ptd_32p.sh
```bash
bash examples/llama/pretrain_llama_33B_ptd_32p.sh
```

Launch llama-65B pre-training script : ModelLink/examples/llama/pretrain_llama_65b_ptd.sh
```bash
bash examples/llama/pretrain_llama_65b_ptd.sh
```
Config llama-33B/65B pre-training script for multinode (Launch llama-65B pre-training script on each machine):

```shell
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=4
NODE_RANK=0
```
The Training log will look like these:

```Shell
 iteration  11/50000 | consumed samples: 5632 | consumed tokens:  11534336 | elapsed time per iteration (ms):  52728.1 | learning rate:    1.499E-05 | gloabl batch size:  512 | lm loss:  1.376514E+01 | loss scale:  65536.0 | grad norm:    459.628 | actual seqlen:  2048 | number of skipped
iterations: 0 | number of nan iterations:   0 | samples per second: 9.710 | TFLOPs: 167.52 |
time (ms)
```

5. Finetune

5.1 Prepare fine-tune dataset
```shell
# for llama, dowload alpaca dataset, like
wget http://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
```

LLaMA-33B
```shell
mkdir dataset
python ./tools/preprocess_data.py \
  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/llama-33b-hf \ 
  --output-prefix ./finetune_dataset/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralInstructionHandler \
  --append-eod
```

LLaMA-65B
```shell
mkdir dataset
python ./tools/preprocess_data.py \
  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/llama-65b-hf \
  --output-prefix ./finetune_dataset/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralInstructionHandler \
  --append-eod
```

5.2 Config llama-33B/65B fine-tune script :
Config llama-33B fine-tune script `./examples/llama/tune_llama_33B_ptd_32p.sh`:
```bash
# modify the script according to your own conda and ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# modify script orign dataset path according to your own dataset path
TOKENIZER_MODEL=./model_from_hf/llama-33b-hf/  #tokenizer path
DATA_PATH=./finetune_dataset/alpaca  #processed dataset
LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

Config llama-65B fine-tune script `./examples/llama/tune_llama_65b_ptd.sh`:
```bash
# modify the script according to your own conda and ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# modify script orign dataset path according to your own dataset path
TOKENIZER_MODEL=./model_from_hf/llama-65b-hf/  #tokenizer path
DATA_PATH=./finetune_dataset/alpaca  #processed dataset
LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
```

5.3 Launch fine-tune script:

Launch llama-33B pre-training script : ModelLink/examples/llama/tune_llama_33B_ptd_32p.sh
```bash
bash examples/llama/tune_llama_33B_ptd_32p.sh
```

Launch llama-65B pre-training script : ModelLink/examples/llama/tune_llama_65b_ptd.sh
```bash
bash examples/llama/tune_llama_65b_ptd.sh
```
Config llama-33B/65B pre-training script for multinode (Launch llama-65B pre-training script on each machine):

```shell
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=4
NODE_RANK=0
```

### Performance

#### Machine performance

The performance of the NPUs in **Ascend** and Reference:

|  Device   |   Model   | throughput rate (tokens/s/p) |
|:---------:|:---------:|:----------------------------:|
| Reference | llama-33B |             776              |
|   NPUs    | llama-33B |             621              |
| Reference | llama-65B |             426              |
|   NPUs    | llama-65B |             348              |


#### Accuracy of the loss

NPU vs GPU loss and relative error:

LLaMa-33B

The NPU runs smoothly, the resource usage is stable, no errors are reported in the middle of the process, the Loss is on a decreasing trend, and the convergence speed is as expected.

![NPU-LOSS](../../sources/images/llama/llama33B_loss.png)

The relative error between NPU and GPU Loss is less than 0.03 throughout, as expected.

![NPU-Relative-Error](../../sources/images/llama/llama33B_relative_error.png)



LLaMa-65B

The NPU runs smoothly, the resource usage is stable, no errors are reported in the middle of the process, the Loss is on a decreasing trend, and the convergence speed is as expected. The relative error between NPU and GPU Loss is less than 0.02 throughout, as expected.

![NPU-Loss-with-weight-and-Relative-Error](../../sources/images/llama/llama65b-shape_layer10_loss-with-weight.png)

## Inference

We support ModelLink Inference for text generation with LLaMA-33B and LLaMA-65B.
Inference different from pre-training, such as we need to Load pre-training checkpoint and the length of the output samples:

Config LLaMA-33B inference script `tasks/inference/generate_llama_33B_ptd.sh`.

Config LLaMA-65B inference script `tasks/inference/generate_llama_65B_ptd.sh`.

```shell
# modify the model weight path and tokenizer path
CHECKPOINT=<checkpoint-path>
VOCAB_FILE=<vocabfile-path>
```

LLaMA-33B:
```shell
bash ./tasks/inference/generate_llama_33B_ptd.sh
```
LLaMA-65B:
```shell
bash ./tasks/inference/generate_llama_65B_ptd.sh
```

Some inference samples are as follows:

LLaMA-33B:

![llama-13B_generate.png](../../sources/images/llama/llama33B_generate.png)

LLaMA-65B:

![llama-65B_generate.png](../../sources/images/llama/llama-65B_generate.png)


## Evaluation with Numerous Benchmarks

We use Boolq benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/boolq).

Config LLaMA-33B evaluation script:

```shell
    CHECKPOINT=./llama-33b-tp4-pp2/
    VOCAB_FILE=./llama-33b-hf/
    # 配置任务和数据路径
    DATA_PATH="./boolq/data/test/"
    TASK="boolq"
    # 配置生成参数
    python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/evaluation/evaluation_llama.py   \
         --task-data-path $DATA_PATH \
         --task $TASK\
         --seq-length 1024 \
         --max-new-tokens 2 \
         --max-position-embeddings 1024 \
         --tensor-model-parallel-size 4  \
         --pipeline-model-parallel-size 2  \
         --num-layers 60 \
         --hidden-size 6656  \
         --ffn-hidden-size 17920 \
         --load ${CHECKPOINT}  \
         --num-attention-heads 52  \
         --tokenizer-type PretrainedFromHF  \
         --tokenizer-name-or-path ${VOCAB_FILE} \
         --tokenizer-not-use-fast \
         --fp16  \
         --micro-batch-size 1  \
         --position-embedding-type rope \
         --normalization RMSNorm \
         --mlp-layer-fusion \
         --seed 42
```

```shell
# start evaluation
# evaluate llama-65B
bash tasks/evaluation/evaluate_llama_65B_ptd.sh
```

The evaluation performance of LLaMA-7B/13B in **Ascend NPU**:

| Task                                           | Model     | NPU  | Benchmark |
|------------------------------------------------|-----------|------|-----------|
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-33B | 83.2 | [83.1](https://paperswithcode.com/sota/question-answering-on-boolq) |
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-65B | 85.7 | [86.6](https://paperswithcode.com/sota/question-answering-on-boolq) |

## Citation

You may also consider original work in your reference:

```shell
@article{Touvron2023llama,
  title={LLaMA: OPen and Efficient Foundation Language Models},
  author={Hugo Touvron*, Thibaut Lavril*, Gautier Izacard*, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal,
  Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave*, Guillaume Lample*},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}}
```