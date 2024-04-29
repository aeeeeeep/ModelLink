#!/bin/bash

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
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
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --task chat \
       --hf-chat-template \
       --add-eos-token '<|eot_id|>' \
       --top-p 0.9 \
       --temperature 0.6 \
       --use-fused-swiglu \
       --use-rotary-position-embeddings \
       --use-fused-rotary-pos-emb \
       --load ${CHECKPOINT}  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --num-layers 32 \
       --hidden-size 4096  \
       --ffn-hidden-size 14336 \
       --position-embedding-type rope \
       --rotary-base 500000 \
       --seq-length 8192 \
       --max-position-embeddings 8192 \
       --max-new-tokens 256 \
       --group-query-attention \
       --num-query-groups 8 \
       --micro-batch-size 1 \
       --num-attention-heads 32  \
       --swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-5 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 16032 \
       --bf16 \
       --seed 42 \
       | tee logs/generate_llama3_8b.log

