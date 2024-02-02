#!/bin/bash
input_dir="/data/models/baichuan2/new_13b/"
output_dir="$input_dir/part_model"
input_quant_dir="/data/models/baichuan2/new_13b_quant"
output_quant_dir="$input_dir/part_model_quant"
output_quant_dir="/home/ctl/models/data_1221_0_10_39_cut_int"
world_size_=2
cut_W_pack_keys_=['W_pack']
cut_row_keys_=['gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']

task_name=${1-inference}
is_quant=${2-0}
use_tiling_copy_stream=1
max_seqence_length=4096

atb_options="ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_OP_BASE_FFTS_MODE_ENABLE=1 HCCL_BUFFSIZE=110"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
atb_stream="ATB_USE_TILING_COPY_STREAM=${use_tiling_copy_stream}"
lccl_options="BACKEND='lccl'"
start_cmd="MAX_SEQ_LEN=$max_seqence_length torchrun --nproc_per_node $world_size_ --master_port 20001 main.py --task $task_name --is_quant $is_quant"
run_cmd="${atb_options} ${atb_async_options} ${atb_stream} ${start_cmd}"

if [[ ${is_quant} -eq 1 ]];then
    if [[ -d "${output_dir}" ]] && [[ -d "${output_quant_dir}" ]];then
        echo "Weight directory exists, running......"
        eval "${run_cmd}"
    elif [[ ! -d "${output_dir}" ]] || [[ ! -d "${output_quant_dir}" ]];then
        echo "Cut quant model weights ......"
        python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ \
                         --cut_W_pack_keys $cut_W_pack_keys_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
        python ./cut_quant_model_util.py --input_path $input_quant_dir --output_path $output_quant_dir
    fi
else
    echo "**********************float model**********************"
     if [[ -d "${output_dir}" ]];then
         echo "Weight directory exists, running......"
         eval "${run_cmd}"
     else
         echo "Cut model weights ......"
         python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ \
                        --cut_W_pack_keys $cut_W_pack_keys_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
     fi
fi



