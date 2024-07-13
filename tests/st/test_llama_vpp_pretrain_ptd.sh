# This is an example: train llama using PTD,
# the number of parameters is not aligned

export LD_LIBRARY_PATH=/usr/local:/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WITHOUT_JIT_COMPILE=1
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_DETERMINISITIC=True
source /usr/local/Ascend/ascend-toolkit/set_env.sh
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6013
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_LOAD_DIR=/home/dataset/ci_engineering/llama2-7B-tp2-pp4-vpp4
CKPT_SAVE_DIR=/home/dataset/ci_engineering/ckpt
LOG_DIR=/home/dataset/ci_engineering/logs

rm -rf $CKPT_SAVE_DIR
rm -rf $LOG_DIR

mkdir $CKPT_SAVE_DIR
mkdir $LOG_DIR
DATA_PATH=/home/dataset/ci_engineering/pretrain_dataset/alpaca_text_document
TOKENIZER_MODEL=/home/dataset/ci_engineering/llama-2-7b-hf/tokenizer.model

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
basepath=$(cd `dirname $0`; cd ../../; pwd)
export PYTHONPATH=${basepath}:$PYTHONPATH
python3.8 -m torch.distributed.launch $DISTRIBUTED_ARGS \
      ${basepath}/pretrain_gpt.py \
      --tensor-model-parallel-size 2 \
      --pipeline-model-parallel-size 4 \
      --sequence-parallel \
      --num-layers 32 \
      --hidden-size 4096 \
      --ffn-hidden-size 11008 \
      --num-attention-heads 32 \
      --tokenizer-type Llama2Tokenizer \
      --tokenizer-model ${TOKENIZER_MODEL} \
      --seq-length 1024 \
      --max-position-embeddings 1024 \
      --micro-batch-size 1 \
      --global-batch-size 8 \
      --make-vocab-size-divisible-by 1 \
      --lr 1.25e-6 \
      --train-iters 2 \
      --lr-decay-style cosine \
      --untie-embeddings-and-output-weights \
      --disable-bias-linear \
      --attention-dropout 0.0 \
      --init-method-std 0.01 \
      --hidden-dropout 0.0 \
      --position-embedding-type rope \
      --normalization RMSNorm \
      --swiglu \
      --no-masked-softmax-fusion \
      --attention-softmax-in-fp32 \
      --min-lr 1.25e-7 \
      --weight-decay 1e-1 \
      --lr-warmup-fraction 0.01 \
      --clip-grad 1.0 \
      --adam-beta1 0.9 \
      --initial-loss-scale 65536 \
      --adam-beta2 0.95 \
      --no-gradient-accumulation-fusion \
      --no-load-optim \
      --no-load-rng \
      --data-path $DATA_PATH \
      --split 100,0,0 \
      --log-interval 1 \
      --save-interval 10000 \
      --eval-interval 1000 \
      --eval-iters 1 \
      --distributed-backend nccl \
      --load $CKPT_LOAD_DIR \
      --save $CKPT_SAVE_DIR \
      --use-deter-comp \
      --num-layers-per-virtual-pipeline-stage 2 \
      --fp16 | tee $LOG_DIR/test_loss.log

exit $?
