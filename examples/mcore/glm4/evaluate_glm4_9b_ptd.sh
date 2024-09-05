#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=2

WORLD_SIZE=$((${NPUS_PER_NODE}*${NNODES}))

DISTRIBUTED_ARGS="--nproc_per_node ${NPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
DATA_PATH="./evaluate/ceval/val/"
TASK="ceval"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun ${DISTRIBUTED_ARGS} evaluation.py   \
       --task-data-path ${DATA_PATH} \
       --task ${TASK} \
       --seq-length 8192 \
       --max-new-tokens 1 \
       --max-position-embeddings 8192 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --num-layers 40  \
       --hidden-size 4096 \
       --ffn-hidden-size 13696 \
       --num-attention-heads 32  \
       --group-query-attention \
       --num-query-groups 2 \
       --disable-bias-linear \
       --add-qkv-bias \
       --swiglu \
       --padded-vocab-size 151552 \
       --make-vocab-size-divisible-by 1 \
       --position-embedding-type rope \
       --use-rotary-position-embeddings \
       --use-partial-rope \
       --rotary-percent 0.5 \
       --load ${CHECKPOINT} \
       --normalization RMSNorm \
       --norm-epsilon 1.5625e-07 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-model ${TOKENIZER_PATH}  \
       --tokenizer-not-use-fast \
       --bf16  \
       --attention-softmax-in-fp32 \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --micro-batch-size 1  \
       --global-batch-size 16 \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --no-rope-fusion \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --untie-embeddings-and-output-weights \
       --no-chat-template \
       --use-mcore-models \
       --seed 42 \
       | tee logs/evaluate_mcore_glm4_9b_${TASK}_ptd.log
