# This is an example: training llama2 using zero/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/HwHiAiUser/src/megatron-core:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH
export HCCL_CONNECT_TIMEOUT=300
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export COMBINED_ENABLE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=16
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH=/runtime/dataset_llama2/llama2_text_document
CHECKPOINT_PATH=/runtime/llama2_ckpt

DS_CONFIG=deepspeed_config_13B.json
ZERO_STAGE=2
GLOBAL_BATCH=32
MICRO_BATCH=2
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
echo logs/llama2_${logfile}.log
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR"

cat <<EOT > $DS_CONFIG
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 32,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "Adam"
    },

    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "allgather_partitions": true,
        "allgather_bucket_size": 1e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 1e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": 1,
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

# Main script
#deepspeed pretrain_llama.py \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       --checkpoint-activations \
       --use-fused-rotary-pos-emb \
       --triangle-attn \
       --DDP-impl local \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 40 \
       --hidden-size 5120 \
       --ffn-hidden-size 13824 \
       --num-attention-heads 40 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path /runtime/tokenizer_llama2/ \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0003 \
       --lr-decay-style cosine \
       --min-lr 3.0e-5 \
       --weight-decay 1.0e-1 \
       --clip-grad 1.0 \
       --lr-warmup-iters 5000 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1.0e-5 \
       --initial-loss-scale 1048576.0 \
       --use-cpu-initialization \
       $ds_args \
       --fp16 > logs/llama2_${logfile}.log 2>&1 &

