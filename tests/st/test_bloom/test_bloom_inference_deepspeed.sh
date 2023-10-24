#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local:/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh

NNODES=1
NPUS_PER_NODE=8

VOCAB_FILE=/home/dataset/llama
basepath=$(cd `dirname $0`; cd ../../..; pwd)
export PYTHONPATH=${basepath}:$PYTHONPATH

ZERO_STAGE=0
MICRO_BATCH_SIZE=1
config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

deepspeed --num_nodes $NNODES --num_gpus $NPUS_PER_NODE \
       ${basepath}/tasks/inference/inference_bloom_pipeline.py \
	   --task 1 2 3 4 5 \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 30  \
       --hidden-size 4096  \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --position-embedding-type alibi \
       --embed-layernorm \
       --tokenizer-type PretrainedFromHF  \
       --load "${CHECKPOINT}"  \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 256 \
       --max-new-tokens 64 \
       --seed 42 \
       --deepspeed \
       --deepspeed_config ${config_json} \
