# This is an example: train llama using PTD,
# the number of parameters is not aligned
# export LD_LIBRARY_PATH=/home/anaconda3/lib:/root/miniconda3/envs/flj/lib:/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
# export INF_NAN_MODE_ENABLE=0

# export MULTI_STREAM_MEMORY_REUSE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

# export HCCL_BUFFSIZE=8
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.170.27.117
MASTER_PORT=6001
NNODES=4
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/home/j00648035/datasets/llama30B/llama_text_document
TOKENIZER_PATH=/home/j00648035/datasets/llama30B/llama/
CHECKPOINT_PATH=./ckpt
rm -rf ./ckpt/*
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# python -m torch.distributed.launch $DISTRIBUTED_ARGS \
# deepspeed --include localhost:0,1,2,3,4,5,6,7 \
# --pipeline-model-parallel-size 4 \
#        --no-pipeline-parallel \
# --use-distributed-optimizer \
# --use-flash-attn \
# --triangle-attn \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       --DDP-impl local \
       --use-contiguous-buffers-in-ddp \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 4 \
       --num-layers 80 \
       --hidden-size 8192 \
       --ffn-hidden-size 22016 \
       --num-attention-heads 64 \
       --micro-batch-size 1 \
       --global-batch-size 256 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 500 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --checkpoint-activations \
       --checkpoint-policy adaptivev2_sche \
       --initial-loss-scale 524288.0 \
       --sequence-parallel \
       --no-async-tensor-model-parallel-allreduce \
       --no-barrier-with-level-1-timing \
       --llama-mlp-layer-fusion \
       --fp16 | tee logs/train.log

#       --deepspeed --deepspeed_config ds_config.json \
#       --checkpoint-activations \