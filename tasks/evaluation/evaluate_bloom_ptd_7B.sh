#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=1
source ./ascend-toolkit/set_env.sh

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer path"
CHECKPOINT="your checkpoint path"
DATA_PATH="your data path"
TASK="your task"


MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# configure generation parameters
python -m torch.distributed.launch $DISTRIBUTED_ARGS tasks/evaluation/evaluation_llama.py   \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 1 \
       --sequence-parallel \
       --num-layers 30 \
       --hidden-size 4096 \
       --load ${CHECKPOINT} \
       --num-attention-heads 32 \
       --padded_vocab_size 250880 \
       --embed-layernorm \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-model ${TOKENIZER_MODEL} \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --vocab-file ${TOKENIZER_MODEL} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --micro-batch-size 1 \
       --global-batch-size 1 \
       --make-vocab-size-divisible-by 1 \
       --attention-dropout 0.0 \
       --init-method-std 0.01 \
       --hidden-dropout 0.0 \
       --position-embedding-type alibi \
       --normalization LayerNorm \
       --use-flash-attn \
       --no-masked-softmax-fusion \
       --attention-softmax-in-fp32 \
       --weight-decay 1e-1 \
       --lr-warmup-fraction 0.01 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --initial-loss-scale 65536 \
       --adam-beta2 0.95 \
       --no-gradient-accumulation-fusion \
       --no-load-optim \
       --no-load-rng \
       --bf16 | tee logs/evaluation.log



