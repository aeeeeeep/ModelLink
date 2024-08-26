#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 多机参数
NPUS_PER_NODE=8
MASTER_ADDR="master node ip"
MASTER_PORT="master node communication port"
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --context-parallel-algo megatron_cp_algo \
    --context-parallel-size 2 \
    --sequence-parallel \
    --num-layers 40 \
    --hidden-size 4096 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 32 \
    --seq-length 32768 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --max-position-embeddings 32768 \
    --padded-vocab-size 151552 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 2 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --overlap-grad-reduce
    --use-partial-rope \
    --rotary-base 5000000 \
    --use-fused-rmsnorm \
    --normalization RMSNorm \
    --swiglu \
    --use-mc2 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --load $CKPT_LOAD_DIR \
    --norm-epsilon 1.5625e-07 \
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
    --no-load-optim \
    --no-load-rng \
    --use-mcore-models \
    --is-instruction-dataset \
    --tokenizer-padding-side left \
    --tokenizer-not-use-fast \
    --no-rope-fusion \
    --finetune \
    --bf16
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

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee logs/tune_mcore_glm4_9b_32k_npu.log
