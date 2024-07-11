#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# MASTER_ADDR="your master node IP"
MASTER_ADDR="90.90.94.170"
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "NODE_RANK ${NODE_RANK}"

# DATA_PATH="your data path"
# TOKENIZER_MODEL="your tokenizer path"
# CKPT_SAVE_DIR="your model save ckpt path"
# CKPT_LOAD_DIR="your model ckpt path"
DATA_PATH='/home/g00679814/Mixtral-8x7B/alpaca_text_document'
TOKENIZER_MODEL='/home/g00679814/model_from_hf/Mixtral-8x7B/tokenizer.model'
CKPT_SAVE_DIR='/home/g00679814/model_weight/pre'
# CKPT_LOAD_DIR="/home/g00679814/model_weight/fa_local_moe_32k_ep2tp4_6_300"
CKPT_LOAD_DIR='~'

TP=4
PP=1
EP=1
CP=2
CP_TYPE='megatron_cp_algo'
# RECOMPUTE_GRANULARITY='full'
# RECOMPUTE_METHOD='block'
# RECOMPUTE_NUM_LAYERS=32
NUM_LAYERS=8

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-permutation-async-comm
"

GPT_ARGS="
    --use-mcore-models  \
    --disable-bias-linear \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --no-position-embedding \
    --vocab-size 32000 \
    --rotary-base 1e6 \

    --no-masked-softmax-fusion \
    --use-fused-rotary-pos-emb \
    --use-flash-attn \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-check-for-nan-in-loss-and-grad \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --no-create-attention-mask-in-dataloader \

   
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE}  \

    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \

    --micro-batch-size 1 \
    --global-batch-size 16 \
    --lr 1e-5 \
    --train-iters 2000 \
    --lr-decay-iters 1280 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2 \
    --clip-grad 1.0 \
    --bf16 \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage  \
"

DATA_ARGS="
    --data-path $DATA_PATH  \
    --split 99990,8,2 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 5001 \
    --eval-iters 100 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
  $MOE_ARGS \
  $GPT_ARGS \
  $DATA_ARGS \
  $OUTPUT_ARGS \
  --distributed-backend nccl \
  | tee logs/train_mixtral.log 
