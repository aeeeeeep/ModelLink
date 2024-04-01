#!/bin/bash
export TOKENIZERS_PARALLELISM=false
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

VOCAB_FILE=/home/dataset/ci_engineering/llama-2-7b-hf
CHECKPOINT_PATH=/home/dataset/ci_engineering/llama2-7B-tp8-pp1
basepath=$(cd `dirname $0`; cd ../../; pwd)
echo $basepath
export PYTHONPATH=${basepath}:$PYTHONPATH

python3.8 -m torch.distributed.launch $DISTRIBUTED_ARGS ${basepath}/inference.py \
       --task greedy do_sample beam_search \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $VOCAB_FILE \
       --tokenizer-not-use-fast \
       --make-vocab-size-divisible-by 1 \
       --swiglu \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 256 \
       --max-new-tokens 64 \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --load $CHECKPOINT_PATH
