export HCCL_BUFFSIZE=110
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_WORKSPPACE_MEM_ALLOC_GLOBAL=1
export ATB_CONTEXT_WORKSPACE_RING=1
export ATB_USE_TILING_COPY_STREAM=1

export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

SCRIPT_DIR=$(cd $(diname $0); pwd)
FLOAT_MODEL_PATH=""
FLOAT_PART_MODEL_PATH=""
QUANT_MODEL_PATH=""
QUANT_PART_MODEL_PATH=""

INPUT_JSON_PATH=""
OUTPUT_JSON_PATH=""

RUNNING_MODE="--run-precision"
# RUNNING_MODE="--run-performance"

DEVICE=0

TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/models/telechat

function fn_run_parallel() {
    echo "running parallel...."
    export QUANT_WEIGHT_PATH="$QUANT_PART_MODEL_PATH"
    python3 -m torch.distributed.run  --nproc_per_node 2 --master_port 25241 run.py \
        --device $DEVICE \
        --input-path $INPUT_JSON_PATH \
        --output-path $OUTPUT_JSON_PATH \
        --checkpoint $FLOAT_PART_MODEL_PATH \
        --run-parallel \
        "$RUNNING_MODE"
}

function fn_run_single() {
    echo "running single...."
    export QUANT_WEIGHT_PATH="$QUANT_MODEL_PATH"
    python3 run.py \
        --device $DEVICE \
        --input-path $INPUT_JSON_PATH \
        --output-path $OUTPUT_JSON_PATH \
        --checkpoint $FLOAT_MODEL_PATH \
        --run-single \
        "$RUNNING_MODE"
}

function fn_cut_float() {
    cp modeling_float_parallel_cut.py $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py
    python handel_weights.py --input-path $FLOAT_MODEL_PATH --output-path $FLOAT_PART_MODEL_PATH --handle-type cut_float
}

function fn_cut_quant() {
    cp modeling_quant_cut.py $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py
    # cut antioutlier float weight
    python handel_weights.py --input-path $FLOAT_MODEL_PATH --output-path $FLOAT_PART_MODEL_PATH --handle-type cut_float
    #cut quant weight
    python handel_weights.py --input-path $QUANT_MODEL_PATH --output-path $QUANT_PART_MODEL_PATH --handle-type cut_quant
}

function fn_main() {
    echo "------- run.sh --------"

    cd $SCRIPT_DIR

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        "--run_performance=0")
            RUNNING_MODE="--run-precision"
            ;;
        "--run_performance=1")
            RUNNING_MODE="--run-performance"
            ;;
        "--run_parallel=0")
            RUN_PARALLEL=0
            ;;
        "--run_parallel=1")
            RUN_PARALLEL=1
            ;;
        "--run_quant=0")
            RUN_QUANT_MODEL=0
            ;;
        "--run_quant=1")
            RUN_QUANT_MODEL=1
            ;;
        "--cut_float")
            fn_cut_float
            ;;
        "--cut_quant")
            fn_cut_quant
            ;;
        "--help")
            echo "bash run.sh --run_performance=0/1 | --run_parallel=0/1 | --run_quant=0/1 | [--cut_float | --cut_quant]"
            ;;
        esac
        shift
    }
    done

    export RUN_QUANT_MODEL=$RUN_QUANT_MODEL

    cp $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py.bak

    if [ $RUN_QUANT_MODEL -eq 1 ]; then
        echo "Copying quant modeling..."
        cp modeling_quant_parallel.py $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py
    else
        echo "Copying float modeling..."
        cp modeling_float_parallel.py $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py
    fi

    if [ $RUN_PARALLEL -eq 1 ]; then
        echo "running on multiple npu..."
        fn_run_parallel
    else
        echo "running on single npu..."
        fn_run_single
    fi

    cp $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py.bak $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py
}

fn_main "$@"
