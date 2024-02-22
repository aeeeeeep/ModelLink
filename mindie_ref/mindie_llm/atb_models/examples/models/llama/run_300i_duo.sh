# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export BIND_CPU=1
export IS_QUANT=0
export MAX_MEMORY_GB=15
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TP_WORLD_SIZE=2
export MASTER_PORT=20030

torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT -m ../../examples.run_pa --model_path $1