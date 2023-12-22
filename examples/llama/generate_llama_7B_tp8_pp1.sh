#!/bin/bash
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4

# modify config according to your own actual situation
CHECKPOINT="your model path"
TOKENIZER_PATH=./llama2-7b-hf/
TOKENIZER_MODEL=./llama2-7b-hf/tokenizer.model

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/inference/inference_llama.py \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --load ${CHECKPOINT} \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${VOCAB_FILE} \
       --tokenizer-model ${TOKENIZER_MODEL}  \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 2048 \
       --max-new-tokens 128 \
       --seed 42 \
       --use-rotary-position-and-embeddings \
       --normalization RMSNorm \
       --exit-on-missing-checkpoint \
       --use-checkpoint-args \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --no-position-embedding \
       --no-masked-softmax-fusion \
       --make-vocab-size-divisible-by 1
