#!/bin/bash
input_dir="/data/models/CodeLlama-34b-Instruct-hf"
output_dir="/data/models/CodeLlama-34b-Instruct-hf_part"
world_size_=2
cut_row_keys_=['q_proj','k_proj','v_proj','gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']

DEVICE=0

TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
if test -d "$output_dir";
then
    echo "Weight directory exists, runing......"
    cp fa_rope/modeling_llama_fa_rope.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py

    # HCCL_OP_BASE_FFTS_MODE_ENABLE=1 ATB_OPERATION_EXECUTE_ASYNC=0 TASK_QUEUE_ENABLE=0 HCCL_BUFFSIZE=110 torchrun --nproc_per_node 2 run_codellama_half_parallel.py --load_path $output_dir
    HCCL_OP_BASE_FFTS_MODE_ENABLE=1 ATB_OPERATION_EXECUTE_ASYNC=0 TASK_QUEUE_ENABLE=0 HCCL_BUFFSIZE=110 \
    torchrun --nproc_per_node 2 run.py --device $DEVICE --checkpoint $output_dir --run-precision --run-parallel
    # torchrun --nproc_per_node 2 run_codellama_half_parallel.py --load_path $output_dir
    # torchrun --nproc_per_node 2 run_llama70b_parallel_performance.py --load_path $output_dir
else
    echo "Cutted Weight directory does not exist, cuting the weight......"
    cp modeling_llama_parallel.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi
