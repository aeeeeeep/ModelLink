# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=20030
export USE_REFACTOR=true

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export HCCL_BUFFSIZE=120
export HCCL_WHITELIST_DISABLE=1
export ATB_CONTEXT_WORKSPACE_RING=1
export ATB_CONTEXT_WORKSPACE_SIZE=2629145600
export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=1
export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=0
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_CONTEXT_WORKSPACE_SIZE=0

extra_param=""
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [ "$USE_REFACTOR" = true ]; then
    extra_param="${extra_param} --use_refactor True"
fi

if [ "$TP_WORLD_SIZE" == "1" ]; then
    python -m examples.run_fa --model_path $1 $extra_param
else
    torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.run_fa --model_path $1 $extra_param
fi