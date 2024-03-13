#!/bin/bash
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=110
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=0
export ATB_CONTEXT_WORKSPACE_RING=1
export ATB_USE_TILING_COPY_STREAM=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export LCCL_ENABLE_FALLBACK=1
export MAX_SEQ_LENGTH=2048
export LONG_SEQ_ENABLE=0

WORLD_SIZE="8"
DEVICE_TYPE="d9"
TASK="run"

script_dir=$(cd $(dirname $0); pwd)
transformers_package_path=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
cp $script_dir/modeling_llama_ascend.py $transformers_package_path/models/llama/modeling_llama.py

function fn_main()
{
    if [[ ! -z "$1" ]];then
        WORLD_SIZE=$1
        echo "[WORLD_SIZE]: $WORLD_SIZE"
        shift
    fi

    if [[ ! -z "$1" ]];then
        DEVICE_TYPE=$1
        echo "[DEVICE_TYPE]: $DEVICE_TYPE"
        shift
    fi

    if [[ ! -z "$1" ]];then
        TASK=$1
        echo "[TASK]: $TASK"
        shift
    fi

    if [[ $DEVICE_TYPE == "d9" ]];then
        export ATB_USE_TILING_COPY_STREAM=0
    fi

    if [[ $TASK == "performance" ]];then
        export TIMEIT=1
    fi

    if [[ $WORLD_SIZE == "1" ]];then
        echo "run model on a single npu"
        python sdk_test.py --task $TASK
    else
        echo "run model on $WORLD_SIZE npus"
        torchrun --nproc_per_node $WORLD_SIZE --master_port 25641 sdk_test.py --task $TASK
    fi
}

fn_main "$@"

