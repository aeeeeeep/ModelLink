#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

DATA_PATH=/home/flj/Megatron_Deepspeed_llama/dataset/enwiki-gpt/gpt_text_sentence
CHECKPOINT_PATH=./ckpt

export LOCAL_RANK=7

python pretrain_llama.py \
       --DDP-impl local \
       --use-contiguous-buffers-in-ddp \
       --num-layers 2 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /home/flj/Megatron_Deepspeed_llama/dataset/gpt2-vocab.json \
       --merge-file /home/flj/Megatron_Deepspeed_llama/dataset/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 | tee logs/train.log
