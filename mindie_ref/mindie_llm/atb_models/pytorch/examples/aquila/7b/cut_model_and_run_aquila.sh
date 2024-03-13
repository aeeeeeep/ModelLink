#!/bin/bash
input_dir="./"
output_dir="./part_model"
cut_c_attn_keys_=['q_proj','k_proj','v_proj']
cut_row_keys_=['gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']

task_name=${1-inference}
world_size_=2
max_seqence_length=4096

atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_OP_BASE_FFTS_MODE_ENABLE=1 HCCL_BUFFSIZE=110"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
lccl_options="BACKEND='lccl'"
start_cmd="MAX_SEQ_LEN=$max_seqence_length torchrun --nproc_per_node $world_size_ --master_port 20002 main.py --task $task_name"
run_cmd="${atb_options} ${atb_async_options} ${start_cmd}"

echo "**********************float model**********************"
if [[ -d "${output_dir}" ]];then
    echo "Weight directory exists, running......"
    eval "${run_cmd}"
else
    echo "Cut model weights ......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ \
                --cut_c_attn_keys $cut_c_attn_keys_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi