#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh

WORLD_SIZE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECK_POINTS=<checkpoints-path>
TOKENIZER_NAME_OR_PATH=<tokenizer-vocab-file-path>

TP_SIZE=8
PP_SIZE=1
python -m torch.distributed.run $DISTRIBUTED_ARGS ./examples/bloom_task/generate_text_bloom.py \
               --load $CHECK_POINTS \
               --tokenizer-type PretrainedFromHF \
               --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH  \
               --tensor-model-parallel-size $TP_SIZE \
               --pipeline-model-parallel-size $PP_SIZE \
               --embed-layernorm \
               --position-embedding-type alibi \
               --num-layers 30 \
               --hidden-size 4096 \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --num-attention-heads 32 \
               --micro-batch-size 1 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --init-method-std 0.0048 \
               --log-interval 1 \
               --layernorm-epsilon 1e-6 \
               --fp16 \
               --no-load-optim \
               --no-load-rng \
               --out-seq-length 1024 \
               --temperature 1.0 \
               --top_p 0.9 \
               --recompute \