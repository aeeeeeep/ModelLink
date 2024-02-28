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
    cd ../../llama
    bash run.sh --performance $world_size_ d3
else
    echo "Cutted Weight directory does not exist, cuting the weight......"
    cp modeling_codellama_cut.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
    python ./cut_weight.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi
