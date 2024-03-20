#!/bin/bash

#
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
CHECKPOINT_LORA="your lora model directory path"
TOKENIZER_PATH="your tokenizer directory path"
TOKENIZER_MODEL="your tokenizer.model file path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/inference/inference_alpaca.py \
       --tensor-model-parallel-size 4  \
       --pipeline-model-parallel-size 2  \
       --num-layers 60 \
       --hidden-size 6656 \
       --ffn-hidden-size 17920 \
       --position-embedding-type rope \
       --seq-length 2048 \
       --max-new-tokens 256 \
       --micro-batch-size 2 \
       --global-batch-size 16 \
       --num-attention-heads 52  \
       --max-position-embeddings 2048 \
       --swiglu \
       --load "${CHECKPOINT}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-model "${TOKENIZER_MODEL}"  \
       --tokenizer-not-use-fast \
       --bf16 \
       --normalization RMSNorm \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --lora-load ${CHECKPOINT_LORA}  \
       --lora-r 16 \
       --lora-alpha 32 \
       --make-vocab-size-divisible-by 1 \
       | tee logs/generate_llama_33b_lora.log
