#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MASTER_ADDR=localhost
MASTER_PORT=6661
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

VOCAB_FILE=/data/gemma-7b-hf
CHECKPOINT_PATH=/data/gemma-7b-tp8-pp1
basepath=$(cd `dirname $0`; cd ../../; pwd)
echo $basepath
export PYTHONPATH=${basepath}:$PYTHONPATH

python3.8 -m torch.distributed.launch $DISTRIBUTED_ARGS ${basepath}/inference.py \
       --task greedy do_sample beam_search \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 28  \
       --hidden-size 3072  \
       --ffn-hidden-size 24576 \
       --num-attention-heads 16  \
       --kv-channels 256 \
       --max-position-embeddings 8192 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${VOCAB_FILE} \
       --tokenizer-not-use-fast \
       --make-vocab-size-divisible-by 1 \
       --geglu \
       --input-embeds-norm \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 8192 \
       --max-new-tokens 64 \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --add-rmsnorm-offset \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --load $CHECKPOINT_PATH
exit $?
