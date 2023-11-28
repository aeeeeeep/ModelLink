# This is an example: training llama using PTD/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib://home/ma-user/anaconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200


# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=12892
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
GLOBAL_BATCH=128
MICRO_BATCH=4
SEQ_LEN=2048

DATA_PATH=/home/ma-user/work/AscendSpeed_wair/wiki
# DATA_PATH=/home/ma-user/work/AscendSpeed_wair/belle_12k_v2
# DATA_PATH=/home/ma-user/work/AscendSpeed/tatsu-lab_alpaca.json
TOKENIZER_PATH=/home/ma-user/work/AscendSpeed/baichuan-13B-hf  
 
CHECKPOINT_PATH=/home/ma-user/work/AscendSpeed/13b_ckpt
LOAD_PATH=/home/ma-user/work/AscendSpeed/13b_weight

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
rm -rf kernel_meta*

# Main script
nohup python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_baichuan.py \
       --DDP-impl local \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --sequence-parallel \
       --num-layers 40 \
       --hidden-size 5120 \
       --ffn-hidden-size 13696 \
       --num-attention-heads 40 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
       --train-iters 1000 \
       --save $CHECKPOINT_PATH \
       --load $LOAD_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 1000,0,0 \
       --make-vocab-size-divisible-by 8 \
       --distributed-backend nccl \
       --lr 1e-5 \
       --lr-decay-style cosine \
       --min-lr 1e-8 \
       --weight-decay 1e-1 \
       --position-embedding-type alibi \
       --clip-grad 1.0 \
       --initial-loss-scale 8188.0 \
       --seed 1234 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1.0e-5 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 10000 \
       --eval-iters 10 \
       --bf16 \
       --tensorboard-dir ./13b_tf_record \
       --dataloader-type cyclic \
       > train_13B.log &
       # --is-instruction-dataset \
       # --dataloader-type cyclic \
       # > train_13B.log &
