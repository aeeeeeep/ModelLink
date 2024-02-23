#!/bin/bash
input_dir="../model_path"
output_dir="$input_dir/part_model"
cut_c_attn_keys_=['attn.c_attn.weight','attn.c_attn.bias']
cut_mlp_keys_=['mlp.w1.weight','mlp.w2.weight']
cut_c_attn_mlp_keys_=['attn.c_proj.weight','mlp.c_proj.weight']

task_name=${1-inference}
rank_size_=4
max_seqence_length=4096
use_launch_kernel_with_tiling=1
atb_operation_execute_async=1
task_queue_enable=1
LONG_SEQ_ENABLE=0

atb_options="ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_OP_BASE_FFTS_MODE_ENABLE=1 HCCL_BUFFSIZE=110"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=${atb_operation_execute_async} TASK_QUEUE_ENABLE=${task_queue_enable}"
atb_launch_kernel="ATB_LAUNCH_KERNEL_WITH_TILING=${use_launch_kernel_with_tiling}"
start_cmd="LONG_SEQ_ENABLE=${LONG_SEQ_ENABLE} MAX_SEQ_LEN=$max_seqence_length torchrun --nproc_per_node $rank_size_ --master_port 20002 main.py --task $task_name"
run_cmd="${atb_options} ${atb_async_options} ${atb_launch_kernel} ${start_cmd}"

echo "**********************float model**********************"
if [[ -d "${output_dir}" ]];then
    echo "Weight directory exists, running......"
    eval "${run_cmd}"
else
    echo "Cut model weights ......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --rank_size $rank_size_ \
                --cut_c_attn_keys $cut_c_attn_keys_ --cut_mlp_keys $cut_mlp_keys_ --cut_c_attn_mlp_keys $cut_c_attn_mlp_keys_
fi