# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TP_WORLD_SIZE=8
export MASTER_PORT=12343
export PYTHONPATH=${llm_path}:$PYTHONPATH

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export LCCL_ENABLE_FALLBACK=1

torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT -m examples.run_pa --model_path $1 $extra_param --max_batch_size 1 --max_input_length 256 --max_output_length 256 --use_refactor