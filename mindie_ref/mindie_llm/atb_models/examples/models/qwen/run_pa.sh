#!/bin/bash
RANK_SIZE=8
master_port=12347
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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

if [[ -n ${model_path} ]];then
    echo ">>>> Load model from ${model_path}"
    if [[ $RANK_SIZE -gt 1 ]]; then
        LD_PRELOAD=/usr/lib64/libgomp.so.1.0.0 torchrun --nproc_per_node ${RANK_SIZE} --master_port ${master_port} ./examples/run_pa.py --model_path ${model_path}
    else
        LD_PRELOAD=/usr/lib64/libgomp.so.1.0.0 python ./examples/run_pa.py --model_path ${model_path}
    fi
fi