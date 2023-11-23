# This is an example: train llama using PTD.
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/HwHiAiUser/src/megatron-core:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH
export HCCL_CONNECT_TIMEOUT=300
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export COMBINED_ENABLE=1

# Change for multinode config
MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=16
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
echo logs/llama2_${logfile}.log

DATA_PATH=/runtime/dataset_llama2/llama2_text_document
CHECKPOINT_PATH=/runtime/llama2_ckpt
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR"

# Main script
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       --DDP-impl local \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --num-layers 40 \
       --hidden-size 5120 \
       --ffn-hidden-size 13824 \
       --num-attention-heads 40 \
       --micro-batch-size 4 \
       --global-batch-size 256 \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --train-iters 10000 \
       --lr-decay-iters 6400 \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path /runtime/tokenizer_llama2/ \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 1.0e-6 \
       --lr-decay-style cosine \
       --min-lr 1.0e-7 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 100 \
       --eval-iters 10 \
       --initial-loss-scale 1048576.0 \
       --checkpoint-activations \
       --triangle-attn \
       --fp16 > logs/llama2_${logfile}.log 2>&1 &
