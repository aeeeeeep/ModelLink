#!/bin/bash
model_name="internlm20b"
bash_path="/home/fengjunyao/model/internlm20b_1208"
model_path="$bash_path/internlm-20b_old"
antioutlier_path="$bash_path/anti"
antioutlier_para_path="$bash_path/anti-paraller"
quant_path="$bash_path/quant"
quant_para_path="$bash_path/quant-paraller"
world_size=2
cut_row_keys=['q_proj','k_proj','v_proj','gate_proj','up_proj']
cut_col_keys=['o_proj','down_proj']

echo "Weight Antioutlier and Quant, running......"
python ./quant.py --model_name=$model_name --model_path=$model_path --quant_output_path=$quant_path \
                  --antioutlier_output_path=$antioutlier_path

if [[ -d "${quant_path}" ]] && [[ -d "${antioutlier_path}" ]];then
    echo "Cut quant model weights ......"
    python ./cut_quant_model_910b_util.py --input_path=$quant_path --output_path=$quant_para_path \
                                          --world_size=$world_size
    echo "Cut antioutlier model weights ......"
    python ./cut_antioutlier_model_util.py --input_path=$antioutlier_path --output_path=$antioutlier_para_path \
                                          --world_size=$world_size --cut_row_keys=$cut_row_keys \
                                          --cut_col_keys=$cut_col_keys
    export INTERNLM_ANTIOUTLIER_WEIGHT_PATH=$antioutlier_para_path
    export INTERNLM_QUANT_WEIGHT_PATH=$quant_para_path
else
    echo "not quant model, exit ......"
    exit 0
fi
