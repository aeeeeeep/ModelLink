# Aquila-7B
[简体中文](https://gitee.com/ascend/ModelLink/blob/master/examples/aquila/README.md) | English

- [Aquila-7B](#aquila-7b)
  - [Training](#training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)
    - [Accuracy of the loss](#accuracy-of-the-loss)
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
# install torch and torch_npu
pip install torch==2.1.0
pip install torch_npu-2.1.0.postxxxx-cp38-cp38-xxxx_aarch64.whl
# install apex
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl
# enter the AscendSpeed/ directory，source proper CANN env file(please modify the path based on your real scenario), then install ascendspeed package by source code
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
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./HF_Aquila7B_downloaded/ \
    --output-prefix ./dataset/aquila \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF
```

5. Weights convert

convert the model pre-training weights.

```shell
mkdir model_weights
cd AscendSpeed
# please modify the path to set_env.sh based on your environment.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py --model-type GPT \
    --load-dir ./HF_Aquila7B_downloaded \
    --save-dir ./model_weights/aquila \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --make-vocab-size-divisible-by 1 \
    --tokenizer-name-or-path ./HF_Aquila7B_downloaded \
    --tokenizer-type PretrainedFromHF
```


### 6. Config Aquila-7B pre-training script.

```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# modify script orign dataset path according to your own dataset path
TOKENIZER_PATH=./HF_Aquila7B_downloaded  #tokenizer path
DATA_PATH=./dataset/aquila_text_document  #processed dataset
CKPT_LOAD_DIR=./model_weights/aquila
CKPT_SAVE_DIR=./ckpt
```
*Note that if you do not load weights for pre-training, you can ignore CKPT_SAVE_DIR, and remove the `--load` parameter from the training script*

### 7. Launch Aquila-7B pre-training script.

start training Aquila-7B model:
```shell
bash examples/aquila/pretrain_aquila_7b_ptd.sh
```
## Performance
### Machine performance
### Accuracy of the loss

Aquila-7B NPU vs Reference loss.
![NPU-GPU-Relative-Error](../../sources/images/aquila/aquila_comp0122.png)


## Inference

We support AscendSpeed Inference for text generation with Aquila 7B model.

Inference is different from pre-training because it requires loading the pre-trained model weights. Therefore, we need to complete the aforementioned model weight conversion task first, then configure the Aquila-7B Inference shell script `examples/aquila/generate_aquila_7B.sh`. "CHECKPOINT" must point to the converted weights directory, and "VOCAB_FILE" must point to the directory which contains Aquila vocabulary files -- in our example, it is "./HF_Aquila7B_downloaded". In your operation, please fill in correct value based on your actual scenario.

```shell
# please change to actual values
CHECKPOINT=<checkpoint-path>
VOCAB_FILE=<vocabfile-path>
```

Start Aquila-7B Inference:
```shell
bash ./examples/aquila/generate_aquila_7B.sh
```

Sample results of Aquila-7B Inference:

![aquila-7B_generate.png](../../sources/images/aquila/aquila_7B_generate.png)


## Evaluation with Benchmark

We use BoolQ benchmark to evaluate our model. You can [go to the BoolQ Benchmark page](https://github.com/google-research-datasets/boolean-questions) and find the [dataset](https://storage.cloud.google.com/boolq/dev.jsonl), download it and save it. For example, save to "AscendSpeed/boolq/test" directory

Evaluation task is similar to inference task，it also requires loading the pre-trained model weights. You can configure Aquila-7B evaluation script as the following example code shows：

```shell
    CHECKPOINT="./model_weights/aquila/"
    VOCAB_FILE="./HF_Aquila7B_downloaded/"
    DATA_PATH="./boolq/test"
    TASK="boolq"
    python -m torch.distributed.launch $DISTRIBUTED_ARGS tasks/evaluation/evaluation_llama.py \
        --task-data-path $DATA_PATH \
        --task $TASK \
        --seq-length 2048 \
        --max-new-tokens 1 \
        --max-position-embeddings 2048 \
        --tensor-model-parallel-size 1  \
        --pipeline-model-parallel-size 1  \
        --num-layers 32  \
        --hidden-size 4096  \
        --ffn-hidden-size 11008 \
        --load ${CHECKPOINT}  \
        --num-attention-heads 32  \
        --tokenizer-type PretrainedFromHF  \
        --tokenizer-name-or-path $VOCAB_FILE \
        --tokenizer-not-use-fast \
        --fp16  \
        --micro-batch-size 1  \
        --position-embedding-type rope \
        --normalization RMSNorm \
        --layernorm-epsilon 1e-6 \
        --make-vocab-size-divisible-by 8 \
        --use-flash-attn \
        --pad-vocab-size-to 100008 \
        --seed 42 | tee logs/train.log
```

```shell
# Start evaluation task
bash examples/aquila/eval_aquila_7B.sh
```

Sample Aquila-7B performance running in **Ascend NPU**:

| Task                                                                   | Model     | NPU | Benchmark |
|------------------------------------------------------------------------|------------|------|------|
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Aquila-7B  | 76.9% |     |
