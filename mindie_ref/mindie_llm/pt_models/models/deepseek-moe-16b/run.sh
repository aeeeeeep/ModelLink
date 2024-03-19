#!/bin/bash
declare -A map
# map["device id"]="numa node id"
map["0"]="0"
map["1"]="0"
map["2"]="0"
map["3"]="0"
map["4"]="1"
map["5"]="1"
map["6"]="1"
map["7"]="1"

export HCCL_BUFFSIZE=120
export HCCL_WHITELIST_DISABLE=1
export TASK_QUEUE_ENABLE=1
export LCCL_ENABLE_FALLBACK=1
unset ASCEND_GLOBAL_LOG_LEVEL

RANK_ID_START=0
WORLD_SIZE=8
weight_dir="/path/to/deepseek-moe/"

MODELING_SCRIPT_PATH="modeling_deepseek_pipe_parallel.py"

echo "Weight directory exists, runing......"
echo "$MODELING_SCRIPT_PATH"
echo "$weight_dir/modeling_deepseek.py"
cp $MODELING_SCRIPT_PATH $weight_dir/modeling_deepseek.py
cp pipeBaseModule.py $weight_dir/pipeBaseModule.py

for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++));
do
export LOCAL_RANK=$RANK_ID
export RANK=$RANK_ID
export WORLD_SIZE=$WORLD_SIZE
bind=${map["$RANK_ID"]}
echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
numactl --cpunodebind=$bind --membind $bind python3 run_deepseek.py --load_path $weight_dir &

done
wait
