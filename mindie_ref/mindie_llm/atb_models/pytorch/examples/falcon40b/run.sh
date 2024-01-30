#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_BUFFSIZE=110

# export ATB_LOG_LEVEL=FATAL      # INFO
# export ATB_LOG_TO_STDOUT=0
# export ATB_LOG_TO_FILE=0        # 按 PID 输出日志到文件
# export ATB_SAVE_TENSOR=0        # 保存 tensor
# # export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1

# export ASDOPS_LOG_LEVEL=FATAL   # INFO
# export ASDOPS_LOG_TO_STDOUT=0
# export ASDOPS_LOG_TO_FILE=0

# 同步操作
export ASCEND_LAUNCH_BLOCKING=0

# default 120s
export HCCL_CONNECT_TIMEOUT=6000

# speed up!!!
export ATB_OPERATION_EXECUTE_ASYNC=1

# export ASCEND_GLOBAL_LOG_LEVEL=1      # 环境变量控制日志级别, 0=debug 1=info 3=error
# export ASCEND_SLOG_PRINT_TO_STDOUT=0  # 日志重定向到stdout, 0=闭, 1=开 

# export ATB_PROFILING_ENABLE=1  # 打开 profiling


run_mode="None"
world_size=4
huggingface_weights=""
npu_weights="/home/weights/falcon40b_parallel/"
dataset=""
FILENAME="1_batch_performance_falcon40b.csv"

while getopts m:h:n:d:f:w: opt_name
do
    case $opt_name in
        m) echo "[+] Setting mode..."
            if [ "$OPTARG" == "cut" -o $OPTARG == "speed" -o $OPTARG == "mmlu" ]; then
                run_mode=$OPTARG
                echo "[+] Mode: $run_mode"
            else
                echo "[!] Invalid mode, expected cut, speed, or mmlu but found $OPTARG."
                exit 1
            fi
            ;;
        h) huggingface_weights=$OPTARG ;;
        n) npu_weights=$OPTARG ;;
        d) dataset=$OPTARG ;;
        f) FILENAME=$OPTARG ;;
        w) echo "[~] Set world_size=$OPTARG"
            world_size=$OPTARG ;;
        :) echo "Invalid option"
            exit 1
            ;;
    esac
done

TRANSFORMER_PACKAGE_PATH=$(python -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/models/falcon
cp modeling_falcon_npu.py $TRANSFORMER_PACKAGE_PATH/modeling_falcon.py

if [ $run_mode == "cut" ]; then
    cp modeling_falcon_cut.py $TRANSFORMER_PACKAGE_PATH/modeling_falcon.py
    python ./cut_falcon.py --input_path $huggingface_weights --output_path $npu_weights --world_size $world_size
fi

if [ $run_mode == "speed" ]; then
    torchrun --nproc_per_node $world_size --master_port 33693 falcon_speed.py --model_path $npu_weights --file_name $FILENAME > run.log
fi

if [ $run_mode == "mmlu" ]; then
    torchrun --nproc_per_node $world_size --master_port 33693 run_falcon_mmlu.py --model_path $npu_weights --dataset_path $dataset
fi
