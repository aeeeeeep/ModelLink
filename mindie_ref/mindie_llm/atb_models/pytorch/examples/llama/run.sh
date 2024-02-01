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

WORLD_SIZE="1"
DEVICE_TYPE="d3"

# customized parameters
input_dir="./llama2-7b_parallel"
device_id=0
multi_batch_size=[1,4,8,16,32]

# single case inference
seqlen_in=128
seqlen_out=128

# single case inference(0) or multi case inference(1)
multi_case=0
# LLAMA2-7B or LLAMA2-13B, use as file name when running multi case inference
model_name="LLAMA2-7B"
seqlen_in_range=[5,11]
seqlen_out_range=[5,11]
set_case_pair=0
seqlen_in_pair=[256,256,512,1024]
seqlen_out_pair=[64,256,512,1024]

ceval_batch_size=1

script_dir=$(cd $(dirname $0); pwd)
transformers_package_path=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')

# if model has already been cutted, then run the model; if not, cut the model first
function fn_main()
{
    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
        echo "[RUN_OPTION]: $RUN_OPTION"
        shift
    fi

    if [[ ! -z "$1" ]];then
        WORLD_SIZE=$1
        echo "[WORLD_SIZE]: $WORLD_SIZE"
        shift
    fi

    if [[ ! -z "$1" ]];then
        DEVICE_TYPE=$1
        if [[ $DEVICE_TYPE == "d9" ]];then
            echo "[DEVICE_TYPE]: 910"
            export ATB_USE_TILING_COPY_STREAM=0
        else
            echo "[DEVICE_TYPE]: 310"
        fi
        shift
    fi

    case "${RUN_OPTION}" in
    "--performance")
        cp $script_dir/modeling_llama_ascend.py $transformers_package_path/models/llama/modeling_llama.py
        if [[ $WORLD_SIZE == "1" ]];then
            echo "run model on a single npu"
            python main.py \
            --load_path $input_dir \
            --world_size $WORLD_SIZE \
            --device $device_id \
            --seqlen_in $seqlen_in \
            --seqlen_out $seqlen_out \
            --multi_case $multi_case \
            --model_name $model_name \
            --multi_batch_size $multi_batch_size \
            --set_case_pair $set_case_pair \
            --seqlen_in_range $seqlen_in_range \
            --seqlen_out_range $seqlen_out_range \
            --seqlen_in_pair $seqlen_in_pair \
            --seqlen_out_pair $seqlen_out_pair
        else
            echo "run model on $WORLD_SIZE npus"
            torchrun --nproc_per_node $WORLD_SIZE --master_port 25641 main.py \
            --load_path $input_dir \
            --world_size $WORLD_SIZE \
            --device $device_id \
            --seqlen_in $seqlen_in \
            --seqlen_out $seqlen_out \
            --multi_case $multi_case \
            --model_name $model_name \
            --multi_batch_size $multi_batch_size \
            --set_case_pair $set_case_pair \
            --seqlen_in_range $seqlen_in_range \
            --seqlen_out_range $seqlen_out_range \
            --seqlen_in_pair $seqlen_in_pair \
            --seqlen_out_pair $seqlen_out_pair
        fi
        ;;

    "--precision")
        cp $script_dir/modeling_llama_ascend.py $transformers_package_path/models/llama/modeling_llama.py
        torchrun --nproc_per_node 2 --master_port 25641 ceval_test.py \
        --load_path $input_dir \
        --device $device_id \
        --batch_size $ceval_batch_size
        ;;
    
    "--help")
        echo "run.sh [--performance|--precision] [world_size] [d3|d9]"
        ;;
        
    *)
        echo "unknown build type:${RUN_OPTION}"
        echo "run.sh [--performance|--precision] [world_size] [d3|d9]"
        ;;
    esac
}

fn_main "$@"
