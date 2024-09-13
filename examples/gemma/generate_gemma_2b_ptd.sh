#!/bin/bash

# The number of parameters is not aligned
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WITHOUT_JIT_COMPILE=1

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=2
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --load ${CHECKPOINT}  \
       --num-layers 18 \
       --hidden-size 2048  \
       --kv-channels 256 \
       --group-query-attention \
       --num-query-groups 1 \
       --ffn-hidden-size 16384 \
       --num-attention-heads 8  \
       --position-embedding-type rope \
       --seq-length 8192 \
       --max-position-embeddings 8192 \
       --max-new-tokens 256 \
       --geglu \
       --input-embeds-norm \
       --micro-batch-size 1 \
       --norm-epsilon 1e-06 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --normalization RMSNorm \
       --add-rmsnorm-offset \
       --disable-bias-linear \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --attention-softmax-in-fp32 \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 1 \
       --vocab-size 256000 \
       --bf16 \
       --seed 42 \
       | tee logs/generate_gemma_2b.log
