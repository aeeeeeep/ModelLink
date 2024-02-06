#!/bin/bash
bash_path="/home/fengjunyao/model/internlm20b_1208"
model_path="$bash_path/internlm-20b_old"
antioutlier_path="$bash_path/anti"
antioutlier_para_path="$bash_path/anti-paraller"
quant_path="$bash_path/quant"
quant_para_path="$bash_path/quant-paraller"
world_size=2

export INTERNLM_ANTIOUTLIER_WEIGHT_PATH=$antioutlier_para_path
export INTERNLM_QUANT_WEIGHT_PATH=$quant_para_path