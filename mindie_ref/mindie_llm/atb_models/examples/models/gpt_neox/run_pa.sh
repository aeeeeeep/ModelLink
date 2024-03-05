#!/bin/bash
export BIND_CPU=1
export IS_QUANT=0
export MAX_MEMORY_GB=30
export ASCEND_RT_VISIBLE_DEVICES=4,5
export MASTER_PORT=20014
export TP_WORLD_SIZE=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))
model_path=$1

atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_BUFFSIZE=110"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
base_cmd="torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT -m examples.run_pa --model_path $model_path"
run_cmd="${atb_options} ${atb_async_options} ${base_cmd}"

if [[ -n ${model_path} ]];then
    eval "${run_cmd}"
fi