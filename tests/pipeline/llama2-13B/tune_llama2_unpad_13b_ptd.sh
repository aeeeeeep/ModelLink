#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export AZUREML_EXPERIMENT_ID=0

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0

cd /usr/local/Ascend/atb
source set_env.sh
cd -

export WITHOUT_JIT_COMPILE=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6021
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_LOAD_DIR=/home/dataset/llama2-13B-tp8-pp1
DATA_PATH=/home/dataset/tune-dataset-llama2-13B/alpaca
TOKENIZER_MODEL=/home/dataset/llama2-13B
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
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 13824 \
    --load ${CKPT_LOAD_DIR} \
    --num-attention-heads 40 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --tokenizer-not-use-fast \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1e-6 \
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
    --min-lr 1e-8 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --finetune \
    --is-instruction-dataset \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --use-unpad \
    --distributed-backend nccl 2>&1 | tee /home/dataset/tune_llama2_13b_ptd_unpad.log