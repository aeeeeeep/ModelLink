rm -rf ~/atb/log/
#!/bin/bash
# input_dir="/home/data/acltransformer_testdata/weights/Mixtral-8x7B-v0.1"

# 绑核以实现性能优化，实现详见readme
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

# 环境变量
export HCCL_BUFFSIZE=120
export HCCL_WHITELIST_DISABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export ATB_LAUNCH_KERNEL_WITH_TILING=1
# export ATB_SAVE_TENSOR=0  #是否开启落地Tensor功能
# export ATB_SAVE_TENSOR_START=0 #落tensor起始次数
# export ATB_SAVE_TENSOR_END=2 #落tensor结束次数
# # export ATB_SAVE_TENSOR_RUNNER="" #runner过滤名字
# export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0 #每个Kernel的Execute时就做同步
# export ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0 #每个Runner的Execute时就做同步
# export ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0 #每个Operation的Execute时就做同步
# export ATB_OPSRUNNER_SETUP_CACHE_ENABLE=1 #是否开启SetupCache，当检查到输入和输出没有变化时，不做setup
# export ATB_OPSRUNNER_KERNEL_CACHE_TYPE=2 #0:不开启, 1:开启所有OpsRunner共享全局缓存, 2:开启每个OpsRunner私有缓存
# export ATB_OPSRUNNER_KERNEL_CACHE_COUNT=1 #缓存个数
# export ATB_OPSRUNNER_KERNEL_CACHE_TILING_SIZE=10240 #tiling默认大小
export ATB_PROFILING_ENABLE=1 #是否开启profiling工具
export ATB_OPERATION_EXECUTE_ASYNC=1 # Operation是否异步运行
export TASK_QUEUE_ENABLE=1
export ATB_CONTEXT_WORKSPACE_RING=1
# export HCCL_MTE_ENABLE=1
# export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export LCCL_ENABLE_FALLBACK=1
# unset LCCL_ENABLE_FALLBACK
export ATB_CONTEXT_WORKSPACE_SIZE=2629145600
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 # MASTER 分支 wenjie mem allocate reduce
# export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=2 # MASTER 分支 suiweiyi mem allocate reduce
unset ASCEND_GLOBAL_LOG_LEVEL
unset ATB_LOG_LEVEL #日志级别
unset ATB_LOG_TO_STDOUT #日志是否输出到控制台
unset ATB_LOG_TO_FILE   #日志是否输出到文件
unset ASCEND_SLOG_PRINT_TO_STDOUT
# export ATB_LOG_LEVEL=INFO
# export ATB_LOG_TO_FILE=1
# # export ATB_LOG_TO_STDOUT=1
# export ASDOPS_LOG_LEVEL=INFO
# export ASDOPS_LOG_TO_FILE=1
# export ASDOPS_LOG_TO_STDOUT=1
# export ATB_LOG_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=3
# export ASCEND_SLOG_PRINT_TO_STDOUT=1

RANK_ID_START=0

# 多卡并行数
WORLD_SIZE=8

# 权重切分数
world_size_=8

# 权重路径，需适配不同机器进行修改
weight_dir="/home/data/acltransformer_testdata/weights/deepseek-moe-16b-chat-8"

# 权重切分方式，行切，列切
cut_row_keys_=['q_proj','k_proj','v_proj','w1','w3']
cut_col_keys_=['o_proj','w2']

# 环境原生transformers库路径，用以侵入式修改modeling文件
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
MODELING_SCRIPT_PATH="$ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/patches/modeling_deepseek_atb.py"

# 性能打点文件侵入式修改
UTILS_PATH="$TRANSFORMER_PACKAGE_PATH/generation/utils.py"
PERSONAL_UTILS_PATH="$ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/utils.py"

# 模型配置文件修改
cp $PERSONAL_UTILS_PATH $UTILS_PATH

cp $MODELING_SCRIPT_PATH $weight_dir/part_model/0/modeling_deepseek.py
cp $MODELING_SCRIPT_PATH $weight_dir/part_model/1/modeling_deepseek.py
cp $MODELING_SCRIPT_PATH $weight_dir/part_model/2/modeling_deepseek.py
cp $MODELING_SCRIPT_PATH $weight_dir/part_model/3/modeling_deepseek.py
cp $MODELING_SCRIPT_PATH $weight_dir/part_model/4/modeling_deepseek.py
cp $MODELING_SCRIPT_PATH $weight_dir/part_model/5/modeling_deepseek.py
cp $MODELING_SCRIPT_PATH $weight_dir/part_model/6/modeling_deepseek.py
cp $MODELING_SCRIPT_PATH $weight_dir/part_model/7/modeling_deepseek.py

cp $ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/config.json $weight_dir/part_model/0/config.json
cp $ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/config.json $weight_dir/part_model/1/config.json
cp $ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/config.json $weight_dir/part_model/2/config.json
cp $ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/config.json $weight_dir/part_model/3/config.json
cp $ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/config.json $weight_dir/part_model/4/config.json
cp $ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/config.json $weight_dir/part_model/5/config.json
cp $ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/config.json $weight_dir/part_model/6/config.json
cp $ATB_SPEED_HOME_PATH/pytorch/examples/deepseek16B_densed/config.json $weight_dir/part_model/7/config.json

# 权重已存在，执行推理
if test -d "$weight_dir";
then
    echo "Weight directory exists, runing......"
    for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++));
    do
    export LOCAL_RANK=$RANK_ID
    export WORLD_SIZE=$WORLD_SIZE
    bind=${map["$RANK_ID"]}
    echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
    numactl --cpunodebind=$bind --membind $bind python3 run_deepseek.py --load_path $weight_dir &
    # numactl --cpunodebind=$bind --membind $bind python3 eval/evaluate_mmlu.py -d data/mmlu/data/ --load_path $weight_dir &
    # numactl --cpunodebind=$bind --membind $bind python3 ./TruthfulQA/main.py --models llama --metrics mc --input_path ./TruthfulQA/TruthfulQA.csv --output_path ./TruthfulQA/TruthfulQA_answers.csv --load_path $weight_dir &

    done
wait 
# 权重不存在，执行切分
else 
    echo "Cutted Weight directory does not exist, cuting the weight......"
    mkdir -p $weight_dir
    python ./cut_model_util.py --input_path $input_dir --output_path $weight_dir --world_size $world_size_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi
