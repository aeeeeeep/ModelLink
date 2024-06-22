#!/bin/bash

# Runs grok1 8x310B model on 256 H100/A100 GPUs
# The Dropless MoE suffers from an imbalanced token distribution at the early stage of training (the first few hundred iterations), which may lead to poor performance and out-of-memory (OOM) issues.
# To check the performance of a Dropless MoE model, we should run the model for at least 500 iterations or resume from trained checkpoints.
export WITHOUT_JIT_COMPILE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=TRUE

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


TOKENIZER_MODEL="your tokenizer path"
DATA_PATH="your data path"
SAVE_PATH="your model save ckpt path"
LOAD_PATH="your model ckpt path"

#recompute paramters
RECOMPUTE_GRANULARITY='full'
RECOMPUTE_METHOD='block'
RECOMPUTE_NUM_LAYERS=1
TRANS_TYPE='local'
CP_TYPE='megatron_cp_algo'


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 8192 # 序列长度配置 32k 32768 16k:16384 4k:4096 8k 8192
    --max-position-embeddings 8192
    --num-layers 1 # 32層
    --hidden-size 6144  #4096 
    --ffn-hidden-size 32768
    --num-attention-heads 48 #32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm #RMSNorm LayerNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-position-embedding
    --use-flash-attn
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 1
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2 #    --moe-grouped-gemm
    --use-distributed-optimizer
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 16
    --lr 1e-5
    --train-iters 5000 # 500000
    --lr-decay-iters 1280 # 320000
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --weight-decay 0.1
    --lr-warmup-iters 2 # 500
    --clip-grad 1.0
    --bf16
    --vocab-size 131072
    --use-distributed-optimizer   
    --no-rope-fusion
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --no-gradient-accumulation-fusion
    --transformer-impl   $TRANS_TYPE 
    --context-parallel-size 2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
)
#10000 1000 10
LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5001 \
    --eval-iters 100 \
    --load ${LOAD_PATH} \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
    )
fi

python3 -m torch.distributed.run  ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} | tee ../logs/npu_grok1.log
