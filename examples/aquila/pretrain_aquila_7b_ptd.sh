#!/bin/bash
source /home/yaojia/yj_canntoolkit_POC.T800.B070/ascend-toolkit/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Please remember to source the CANN set_env.sh file

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="./ckpt"
DATA_PATH="../dataset/aquila_text_document"
TOKENIZER_PATH="../HF_Aquila7B_downloaded"
#CKPT_LOAD_DIR="/home/yaojia/bigfiles/data-loading/GPU/modellink-converted-weights-xlc-div1"
#CKPT_LOAD_DIR="/home/yaojia/bigfiles/data-loading/GPU/HF-div1"
CKPT_LOAD_DIR="/home/yaojia/bigfiles/data-loading/GPU/HF-Aquila7B_converted_div1"
TP=8
PP=1



DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --tokenizer-type PretrainedFromHF \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --norm-epsilon 1e-6 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --tokenizer-not-use-fast \
    --use-flash-attn
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 20 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load $CKPT_LOAD_DIR \
    | tee logs/train-Aquila-7b-ptd.log