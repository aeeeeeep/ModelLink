#!/bin/bash

export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=5400
export HCCL_WHITELIST_DISABLE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:1024"

script_path=$(realpath $0)
main_dir=$(dirname $script_path)
source "${main_dir}/configs/model_glm_130b.sh"

# 请根据机器上的NUMA亲和性配置每个芯片对应的NUMA node映射,并通过numactl进行绑核
# 查询NUMA node命令示例：lspci -vs c1:00.0
declare -A map
map["0"]="3"
map["1"]="3"
map["2"]="2"
map["3"]="2"
map["4"]="0"
map["5"]="0"
map["6"]="1"
map["7"]="1"

RANK_ID_START=0
WORLD_SIZE=8
for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++));
do
    export LOCAL_RANK=$RANK_ID
    export WORLD_SIZE=$WORLD_SIZE
    bind=${map["$RANK_ID"]}
    echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
    numactl --cpunodebind=$bind --membind $bind \
        python3 ${main_dir}/main.py $* ${MODEL_ARGS} &
done
wait
