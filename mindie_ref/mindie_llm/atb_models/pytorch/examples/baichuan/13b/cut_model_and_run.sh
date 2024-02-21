#!/bin/bash
input_dir="/data/models/baichuan/13b/"
output_dir="/data/models/baichuan/13b/baichuan-13b-part-test"
world_size_=2
cut_W_pack_keys_=['W_pack']
cut_row_keys_=['gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']
task_name=${1-inference}
if test -d "$output_dir";
then
    echo "Weight directory exists, running......"
    #ATB_LOG_TO_STDOUT=1 ATB_LOG_LEVEL=FATAL TASK_QUEUE_ENABLE=0 ASDOPS_LOG_TO_STDOUT=1 ASDOPS_LOG_LEVEL=FATAL ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1 
    HCCL_OP_BASE_FFTS_MODE_ENABLE=1 ATB_OPERATION_EXECUTE_ASYNC=0 TASK_QUEUE_ENABLE=0 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 HCCL_BUFFSIZE=110 torchrun --nproc_per_node ${world_size_} --master_port 20001 main.py --task "${task_name}"
else
    echo "Cut Weight directory does not exist, cutting the weight......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ \
           --cut_W_pack_keys $cut_W_pack_keys_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi
