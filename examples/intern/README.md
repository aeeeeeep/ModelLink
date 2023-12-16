# Intern-LM
<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/AscendSpeed/blob/master/examples/intern/README_en.md">English</a> </b> 
    </p>
</p>

# 目录

- [Internlm-7B](#internlm-7b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
      - [精度](#精度)
    - [推理](#推理)
    - [评估](#评估)
- [Internlm-65B](#internlm-65b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
      - [精度](#精度)

# InternLM-7B

## 训练

InternLM-7B 训练的硬件配置如下:

|  硬件 |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |

InternLM-7B 训练的软件配置如下:

|            软件             |                                                      配置                                                      |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------:|
|          python           |                                                    3.8.18                                                    |
|          driver           | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|         firmware          | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|           CANN            |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
| binary arithmetic package |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
|           torch           |                                                    2.1.0                                                    |
|         torch_npu         |                             [package](https://gitee.com/ascend/pytorch/releases)                             |
|         apex         |                             [package](https://gitee.com/ascend/apex/releases)                             |

### 脚本

1. 拷贝仓库到本地服务器
```shell
git clone https://gitee.com/ascend/AscendSpeed.git 
cd AscendSpeed 
mkdir logs
mkdir ckpt
```

2. 搭建环境

```bash
# python3.7
conda create -n test python=3.8
conda activate test
# 安装 torch 和 torch_npu
pip install torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch_npu-2.1.0.post20231124_cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 安装 apex
pip install apex-0.1_ascend_20231124-cp38-cp38-linux_aarch64.whl
# 安装 megatron-core
pip3 install --no-use-pep517 -e git+https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core
# 安装 deepspeed 和 deepspeed_npu
pip install deepspeed==0.9.2
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install -e ./
cd ..
# 安装其余依赖包
pip install -r requirements.txt 
```

3. 下载 Internlm-7B [词表文件](https://huggingface.co/internlm/internlm-7b/tree/main)

```shell
#!/bin/bash
mkdir -p dataset/internlm
cd ./dataset/internlm
wget https://huggingface.co/internlm/internlm-7b/resolve/main/config.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/generation_config.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/special_tokens_map.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenization_internlm.py
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer.model
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer_config.json
cd ../..
```

4. 下载 Internlm-7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

```shell
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./dataset/internlm \
    --output-prefix ./dataset/alpaca \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF  \
    --handler-name AlpacaPretrainHandler  \
    --tokenizer-not-use-fast \
    --append-eod
```

5. 权重格式转换

下载 Internlm-7B [权重](https://huggingface.co/internlm/internlm-7b/tree/main) 

```text
# 请注意，如果要加载huggingface的预训练权重，需要修改一个deepspeed关于加载权重的bug：
# 在 `<deepspeed-installed-path>/runtime/engine.py` 文件里的 `_load_zero_checkpoint` 函数，
# 将 `if zero_sd_list is None` 改为 `if zero_sd_list is None or len(zero_sd_list) == 0`

# 原始 deepspeed/runtime/engine.py, 大概 #Lines2746-2748
zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
if zero_sd_list is None:
    return False

# 修改后
zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
if zero_sd_list is None or len(zero_sd_list) == 0:
    return False
```

```shell
mkdir model_from_hf
cd ./model_from_hf
# 必须安装 git-lfs
git clone https://huggingface.co/internlm/internlm-7b
cd ..
```

将模型权重从 huggingface 格式转换为 AscendSpeed 可以处理的格式
```shell
mkdir model_weights
SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
    --input-model-dir ./model_from_hf/internlm-7b/ \
    --output-model-dir ./model_weights \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --type 7B \
    --bias \
    --deepspeed \
```

6. 配置 Internlm-7B 预训练脚本

```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集，词表，权重等路径
TOKENIZER_PATH=./dataset/internlm  #tokenizer path
DATA=./dataset/alpaca_text_document  #processed dataset
CHECKPOINT=./model_weights/
```

7. 启动 Internlm-7B 预训练脚本

```shell
bash examples/intern/pretrain_internlm_7b_zero.sh 
```


### 性能

#### 吞吐

Internlm-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 | 模型          | 总迭代数 | 样本吞吐 (samples/p/s) | token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 浮点计算数 (TFLOPs/s) |
|----|-------------|------|--------------------|----------------------|-----------------|------------------|
| NPUs | Internlm-7B | 2048 | 13.000              | 3328                 | 19684.6            | 145.69           |
| 参考 | Internlm-7B | - | -              | 4078                 |  -            | -           |



#### 精度

NPU vs 参考 （无预训练权重） loss 对比和相对误差
![NPU-Loss-and-Relative-Error](../../sources/images/intern7b_loss.png)

NPU vs 参考 （有预训练权重） loss 对比和相对误差
![NPU-Loss-with-weight-and-Relative-Error](../../sources/images/intern7b_loss_with_weights.png)

#### 推理
<a href="https://gitee.com/ascend/AscendSpeed/blob/master/examples/intern/generate_internlm_7b_deepspeed.sh">对话</a>：
```
#!/bin/bash

export TOKENIZERS_PARALLELISM=false

NNODES=1
NPUS_PER_NODE=8

CHECKPOINT="Your checkpoint path"
VOCAB_FILE="Your vocab path"

ZERO_STAGE=0
MICRO_BATCH_SIZE=1
config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 8
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

deepspeed --num_nodes $NNODES --num_gpus $NPUS_PER_NODE \
       ./tasks/inference/inference_llama.py \
       --row-col-parallel-linear-bias \
       --no-contiguous-buffers-in-local-ddp \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF  \
       --load "${CHECKPOINT}"  \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --seq-length 2048 \
       --max-new-tokens 64 \
       --seed 42 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --no-pipeline-parallel \
       --position-embedding-type rope \
       --normalization RMSNorm \
```

#### 评估
<a href="https://gitee.com/ascend/AscendSpeed/blob/master/tasks/evaluation/eval_internlm.sh">评估</a>:
```
# This is an example: training llama using PTD/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="Your ckpt file path"
VOCAB_FILE="Your vocab file path"
DATA_PATH="../dataset/mmlu/test ../dataset/gsm8k/test ../dataset/boolq/test ../dataset/ceval/test"
TASK="boolq gsm8k mmlu ceval agieval bbh human_eval"
# Different task needs different max_new_tokens value, please follow the instruction in readme.
python -m torch.distributed.launch $DISTRIBUTED_ARGS tasks/evaluation/evaluation_llama.py   \
       --row-col-parallel-linear-bias \
       --task-data-path $DATA_PATH \
       --task $TASK \
       --seq-length 2048 \
       --max-new-tokens 2 \
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
       --seed 42 | tee logs/train.log
```
# InternLM-65B

## 训练

InternLM-65B 训练的硬件配置如下:

|  硬件 |       配置        |
|:---:|:---------------:|
| NPU | 32 x Ascend NPUs |

InternLM-65B 训练的软件配置如下:

|            软件             |                                                      配置                                                      |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------:|
|          python           |                                                    3.8.18                                                    |
|          driver           | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|         firmware          | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|           CANN            |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
| binary arithmetic package |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
|           torch           |                                                    2.1.0                                                    |
|         torch_npu         |                             [package](https://gitee.com/ascend/pytorch/releases)                             |
|         apex         |                             [package](https://gitee.com/ascend/apex/releases)                             |

### 脚本

1. 拷贝仓库到本地服务器
```shell
git clone https://gitee.com/ascend/AscendSpeed.git 
cd AscendSpeed 
mkdir logs
mkdir ckpt
```

2. 搭建环境

```bash
# python3.7
conda create -n test python=3.8
conda activate test
# 安装 torch 和 torch_npu
pip install torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch_npu-2.1.0.post20231124_cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 安装 apex
pip install apex-0.1_ascend_20231124-cp38-cp38-linux_aarch64.whl
# 安装 megatron-core
pip3 install --no-use-pep517 -e git+https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core
# 安装 deepspeed 和 deepspeed_npu
pip install deepspeed==0.9.2
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install -e ./
cd ..
# 安装其余依赖包
pip install -r requirements.txt 
```

3. 下载 [词表文件](https://huggingface.co/internlm/internlm-7b/tree/main)

```shell
#!/bin/bash
mkdir -p dataset/internlm
cd ./dataset/internlm
wget https://huggingface.co/internlm/internlm-7b/resolve/main/config.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/generation_config.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/special_tokens_map.json
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenization_internlm.py
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer.model
wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer_config.json
cd ../..
```

4. 下载 Internlm-65B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

```shell
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./dataset/internlm \
    --output-prefix ./dataset/alpaca \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF  \
    --handler-name AlpacaPretrainHandler  \
    --tokenizer-not-use-fast \
    --append-eod
```

6. 配置 Internlm-65B 预训练脚本

```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
# 修改数据集，词表，权重等路径
TOKENIZER_PATH=./dataset/internlm  #tokenizer path
DATA=./dataset/alpaca_text_document  #processed dataset
CHECKPOINT=./model_weights/
```

7. 启动 Internlm-65B 预训练脚本

```shell
bash examples/intern/pretrain_internlm_65b_zero.sh 
```


### 性能

#### 吞吐

Internlm-65B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 | 模型          | 总迭代数 | 样本吞吐 (samples/p/s) | token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 浮点计算数 (TFLOPs/s) |
|----|-------------|------|--------------------|----------------------|-----------------|------------------|
| NPUs | Internlm-65B | 50000 | 5.33              | 342                 | 24            | 137.8           |
| Reference | Internlm-65B | - | -              | 414                 | -            | -           |



#### 精度

NPU vs 参考 （无预训练权重） loss 对比和相对误差
![NPU-Loss-and-Relative-Error](../../sources/images/intern65b_loss.png)