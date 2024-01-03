export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export INF_NAN_MODE_ENABLE=1
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TOKENIZER_PATH=./dataset/internlm
DATA=./dataset/alpaca/alapaca_packed_input_ids_document
CHECKPOINT=./model_weights/

DS_CONFIG=ds_config.json
ZERO_STAGE=2
GLOBAL_BATCH=256
MICRO_BATCH=2

cat <<EOT > $DS_CONFIG
{
    "fp16": {
        "enabled": false,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 8,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "bf16": {
        "enabled": true
    },
 
    "optimizer": {
        "type": "Adam"
    },
 
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
 
    "gradient_accumulation_steps": 16,
    "train_batch_size": $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu":$MICRO_BATCH,
    "zero_allow_untested_optimizer": true
}
EOT
 
ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
 
 
deepspeed  pretrain_intern.py \
       --DDP-impl local \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 32 \
       --hidden-size 4096 \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --data-path $DATA \
       --load $CHECKPOINT \
       --tokenizer-name-or-path $TOKENIZER_PATH\
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
       --use-flash-attn \
       --auto-recompute-device-size 46080 \
       --use-fused-rmsnorm \
       --use-fused-rotary-pos-emb \
       $ds_args \
       --bf16 | tee logs/train.log
