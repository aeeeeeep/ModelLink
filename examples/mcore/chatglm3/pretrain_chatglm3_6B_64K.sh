#!/bin/bash
# export WITHOUT_JIT_COMPILE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=<local_rank>
MASTER_PORT=6001
NNODES=2
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))


CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=1
CP=16
MBS=1
GBS=32
SEQ_LEN=65536
CP_ALGO=hybrid_cp_algo

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 28 \
    --hidden-size 4096 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 32 \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --transformer-impl local \
    --context-parallel-algo ${CP_ALGO} \
    --ulysses-degree-in-cp 8 \
    --context-parallel-size ${CP}
    --max-position-embeddings 65536 \
    --padded-vocab-size 65024 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 2 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --no-rope-fusion \
    --use-distributed-optimizer \
    --use-partial-rope \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --swiglu \
    --no-create-attention-mask-in-dataloader \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --lr 1e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
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
    --load ${CKPT_LOAD_DIR}  \
    --no-load-optim \
    --no-load-rng \
    --fp16 \
    --num-workers 1 \
    --kv-head-repeat-before-uly-alltoall \
    --no-shared-storage

"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee logs/train_chatglm3_6B_64K.log
