# Aquila-7B
[简体中文](https://gitee.com/ascend/ModelLink/blob/modellink/examples/aquila/README.md) | English

- [Aquila-7B](#aquila-7b)
  - [Training](#training)
    - [Script](#script)
    - [Performance](#performance)
      - [Machine performance](#machine-performance)
  - [Inference](#inference)
  - [Evaluation](#evaluation-with-benchmark)

## Training

Here's a hardware summary of pre-training Aquila-7B:

| Hardware | Device |      Value       |
|:--------:|:------:|:----------------:|
|   NPU    |  A+K   | 8 x Ascend NPUs  |

### Script

1. Clone the repository to your local server and switch to modellink branch:
```shell
git clone -b modellink https://gitee.com/ascend/ModelLink.git
cd ModelLink
mkdir logs
mkdir ckpt
```


2. Build conda environment

```bash
# python3.8
conda create -n test python=3.8
conda activate test
# install torch, torch_npu and apex
pip install torch==2.1.0
pip install torch_npu-2.1.0.postxxxx-cp38-cp38-xxxx_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl
# use git to clone the AscendSpeed source code, enter the directory, source the set_env.sh file based on your host settings(you may need to change the path), then install ascendspeed package by source code
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -e ./

# enter the ModelLink/ directory and install other packages
cd ModelLink/
pip install -r requirements.txt
```


3. Download the Aquila-7B model, config, and tokenizer from [here](https://huggingface.co/BAAI/Aquila-7B/tree/main)

save to ModelLink/HF_Aquila7B_downloaded/ directory.


4. Prepare dataset.

step1: Download the datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet), save to ModelLink/dataset/ directory.


```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```


step2: use Aquila-7B specified tokenizer to pre-process data:


```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ModelLink/
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./HF_Aquila7B_downloaded \
    --output-prefix ./dataset/aquila \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF
```

5. Weights convert

HuggingFace weights --> Megatron weights
***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

```shell
cd ModelLink
mkdir model_weights
# please modify the path to set_env.sh based on your environment.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/util.py \
    --model-type GPT \
    --load-dir ./HF_Aquila7B_downloaded \
    --save-dir ./model_weights/aquila \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --tokenizer-model ./HF_Aquila7B_downloaded/tokenizer.json
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
    --load-dir ../HF_Aquila7B-v0.1-pt8-pp1 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ../HF_Aquila7B_downloaded   # <-- Fill in the original HF model path here, new weights will be saved in ../HF_Aquila7B_downloaded/mg2hg
```


6. Config Aquila-7B pre-training script.

Config the environment variables in aquila pretrain script 
```shell
# set dataset path, CKPT load path for loading weights, and the tokenizer path
TOKENIZER_PATH=./HF_Aquila7B_downloaded  #tokenizer path
DATA_PATH=./dataset/aquila_text_document  #processed dataset
CKPT_LOAD_DIR=./model_weights/aquila   # pointing to the converted model weights
CKPT_SAVE_DIR=./ckpt                   # pointing to the path to save checkpoints
```
*Note that if you do not load weights for pre-training, you can ignore CKPT_SAVE_DIR, and remove the `--load` parameter from the training script*

7. Launch Aquila-7B pre-training script.

Before running the pre-training script, please execute the set_env.sh script first to setup environment variables. Alternatively, you can do this inside aquila pre-training script.

```shell
# you may need to change the path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Start pre-training Aquila-7B model:
```shell
bash examples/aquila/pretrain_aquila_7b_ptd.sh
```
### Performance
#### Machine performance
The performance of Aquila-7B in Ascend NPU and reference device:

| Device | Hardware           | Model       | Iterations | throughput rate (tokens/p/s) | single iteration step time (s/step) |
|------|---------------|------------|------|------------------------|----------------------|
| NPU  | 910b 1node*8p | Aquila-7B  | 1000 | 2849                  | 5.75                  |
| Reference  |              | Aquila-7B  | 1000 | 2874                   |    5.70               |



## Inference

We support AscendSpeed Inference for text generation with Aquila 7B model.

Inference is different from pre-training because it requires loading the pre-trained model weights. Therefore, we need to complete the aforementioned model weight conversion task first, then configure the Aquila-7B Inference shell script `tasks/inference/generate_aquila_7b_ptd.sh`. "CKPT_LOAD_DIR" must point to the converted weights directory, and "TOKENIZER_PATH" must point to the directory which contains Aquila vocabulary files -- in our example, it is "./HF_Aquila7B_downloaded". In your operation, please fill in correct value based on your actual scenario.

```shell
# please change to actual values
CKPT_LOAD_DIR="./model_weights/aquila/"
TOKENIZER_PATH="./HF_Aquila7B_downloaded/"
```

Start Aquila-7B Inference:
```shell
bash ./tasks/inference/generate_aquila_7b_ptd.sh
```

Sample results of Aquila-7B Inference:

![aquila-7B_generate.png](../../sources/images/aquila/aquila_7B_generate.png)


## Evaluation with Benchmark

We use BoolQ benchmark to evaluate our model. You can [go to the BoolQ Benchmark page](https://github.com/google-research-datasets/boolean-questions) and find the [dataset](https://storage.cloud.google.com/boolq/dev.jsonl), download it and save it. For example, save to "ModelLink/boolq/test" directory

Evaluation task is similar to inference task too，it also requires loading the pre-trained model weights. Please note that the model structure parameters used in converting weights should be consistent with those used in running the evaluation task.

After weight conversion is complete, we configure the Aquila-7B evaluation script `tasks/evaluation/evaluate_aquila_7b_ptd.sh`. We need to correctly specify the path to load weights, the path to tokenizer and vocab, and so on (the following example is for reference only)

```shell
    CKPT_LOAD_DIR="./model_weights/aquila/"
    TOKENIZER_PATH="./HF_Aquila7B_downloaded/"
    EVAL_DATA_PATH="./boolq/test"
    TASK="boolq"
```

Start evaluation task
```shell
bash ./tasks/evaluation/evaluate_aquila_7b_ptd.sh
```

Sample Aquila-7B performance running in **Ascend NPU**:

| Task                                                                   | Model     | NPU | Benchmark |
|------------------------------------------------------------------------|------------|------|------|
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Aquila-7B  | 77.3% |     |
