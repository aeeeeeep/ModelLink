#!/bin/bash
source /home/yaojia/yj_canntoolkit_POC.T800.B070/ascend-toolkit/set_env.sh

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CKPT_LOAD_DIR="/home/yaojia/bigfiles/data-loading/GPU/HF-Aquila7B_converted_div1"
#CKPT_LOAD_DIR="/home/yaojia/bigfiles/data-loading/GPU/modellink-converted-weights-xlc"
#CKPT_LOAD_DIR="/home/yaojia/bigfiles/data-loading/GPU/HF-div1"
TOKENIZER_PATH="../HF_Aquila7B_downloaded"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/inference/inference_llama.py \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32 \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --position-embedding-type rope \
       --seq-length 2048 \
       --max-new-tokens 256 \
       --micro-batch-size 8 \
       --global-batch-size 64 \
       --num-attention-heads 32  \
       --max-position-embeddings 4096 \
       --swiglu \
       --load "${CKPT_LOAD_DIR}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --bf16 \
       --normalization RMSNorm \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --norm-epsilon 1e-6 \
       --use-fused-rmsnorm \
       --make-vocab-size-divisible-by 1