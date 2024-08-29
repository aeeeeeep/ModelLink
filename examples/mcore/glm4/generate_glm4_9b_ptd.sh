#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6003
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=2
WORLD_SIZE=$((${NPUS_PER_NODE}*${NNODES}))

DISTRIBUTED_ARGS="--nproc_per_node ${NPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

torchrun ${DISTRIBUTED_ARGS} inference.py \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --num-layers 40  \
       --hidden-size 4096  \
       --ffn-hidden-size 13696 \
       --seq-length 8192 \
       --group-query-attention \
       --num-query-groups 2 \
       --num-attention-heads 32  \
       --padded-vocab-size 151552 \
       --make-vocab-size-divisible-by 1 \
       --max-position-embeddings 8192 \
       --position-embedding-type rope \
       --use-rotary-position-embeddings \
       --use-partial-rope \
       --disable-bias-linear \
       --add-qkv-bias \
       --swiglu \
       --norm-epsilon 1.5625e-07 \
       --hidden-dropout 0.0 \
       --attention-dropout 0.0 \
       --normalization RMSNorm \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --global-batch-size 16 \
       --load ${CHECKPOINT}  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --untie-embeddings-and-output-weights \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --use-mcore-models \
       --no-rope-fusion \
       --seed 42 \
       --bf16 \
       | tee logs/generate_mcore_glm4_9b_ptd.log
