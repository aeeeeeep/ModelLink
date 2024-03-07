# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export BIND_CPU=1
export IS_QUANT=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export TP_WORLD_SIZE=2
export MASTER_PORT=20030

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export TASK_QUEUE_ENABLE=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1

export HCCL_BUFFSIZE=110
export ATB_USE_TILING_COPY_STREAM=1

export PYTHONPATH=${llm_path}:$PYTHONPATH
torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT -m examples.run_pa --model_path $1