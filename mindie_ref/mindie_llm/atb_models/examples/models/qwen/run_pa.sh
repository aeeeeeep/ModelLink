#!/bin/bash
export BIND_CPU=1
export IS_QUANT=0
export RESERVED_MEMORY_GB=3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12347
export TP_WORLD_SIZE=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))
model_path=""

function usage(){
    echo "$0 pls. use '-m|--model-path' input model path"
    exit -1
}

if [[ $# -eq 0 ]];then
        usage
fi

GETOP_ARGS=`getopt -o m: -al model-path:: -- "$@"`
eval set -- "${GETOP_ARGS}"
while [ -n "$1" ]
do
    case "$1" in
        -m|--model-path) model_path=$2;shift 2;;
        --) shift;break;;
        *) usage;break;;
    esac
done

atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_BUFFSIZE=110"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
base_cmd="torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT -m examples.run_pa --model_path $model_path"
run_cmd="${atb_options} ${atb_async_options} ${base_cmd}"

if [[ -n ${model_path} ]];then
    eval "${run_cmd}"
fi