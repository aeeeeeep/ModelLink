#!/bin/bash

# Please remember to source the CANN set_env.sh file
source /home/yaojia/yj_canntoolkit_POC.T800.B070/ascend-toolkit/set_env.sh

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
CKPT_LOAD_DIR="/home/yaojia/bigfiles/data-loading/GPU/HF-Aquila7B_converted_div1"
TOKENIZER_PATH="/home/yaojia/bigfiles/data-loading/GPU/HF_Aquila7B_downloaded"
TP=8
PP=1
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

python -m torch.distributed.launch $DISTRIBUTED_ARGS tasks/inference/inference_llama.py \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 32 \
       --hidden-size 4096 \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32 \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-type PretrainedFromHF \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --micro-batch-size 1 \
       --norm-epsilon 1e-6 \
       --make-vocab-size-divisible-by 1 \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --use-fused-rmsnorm \
       --swiglu \
       --no-masked-softmax-fusion \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --exit-on-missing-checkpoint \
       --max-new-tokens 256 \
       --load $CKPT_LOAD_DIR \
       --tokenizer-not-use-fast \
       --no-gradient-accumulation-fusion \
       --bf16 | tee logs/generate_aquila_7b_ptd.log