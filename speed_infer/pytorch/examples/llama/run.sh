#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 如何使用 run.sh
# bash run.sh --performance   测试性能,运行的是run_llama_performance.py
# bash run.sh --precision     测试准确度,运行的是run_llama_precision.py
# bash run.sh --run           对话例子,运行的是run_llama_example.py
# 兼容模型 llama1-7b llama1-13b llama2-7b llama2-13b，只需指定对应的权重路径即可
# 第一个参数是选择入口脚本 RUN_OPTION_LIST

set -e
MODEL_TARGET_DIR=$(cd $(dirname $0); pwd)
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --precision"
MODEL_PATH="/data/acltransformer_testdata/weights/llama/llama2-7b"
DEVICE_ID=0
NPU=910

function fn_modeling_prepare_310()
{
    if [[ $RUN_OPTION == "--run" ]];then
        cp $MODEL_TARGET_DIR/modeling_llama_310.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
        echo "modeling_llama_310.py copy success"
    elif [[ $RUN_OPTION == "--performance" ]];then
        cp $MODEL_TARGET_DIR/modeling_llama_310.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
        echo "modeling_llama_310.py copy success"
    elif [[ $RUN_OPTION == "--precision" ]];then
        cp $MODEL_TARGET_DIR/modeling_llama_precision_310.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
        echo "modeling_llama_precision_310.py copy success"
    else
        echo "modeling_llama.py unchanged"
    fi
    cd $MODEL_TARGET_DIR
}

function fn_modeling_prepare_910()
{
    if [[ $RUN_OPTION == "--run" ]];then
        cp $MODEL_TARGET_DIR/modeling_llama_910.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
        echo "modeling_llama_910.py copy success"
    elif [[ $RUN_OPTION == "--performance" ]];then
        cp $MODEL_TARGET_DIR/modeling_llama_910.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
        echo "modeling_llama_precision_910.py copy success"
    elif [[ $RUN_OPTION == "--precision" ]];then
        cp $MODEL_TARGET_DIR/modeling_llama_precision_910.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
        echo "modeling_llama_precision_910.py copy success"
    else
        echo "modeling_llama.py unchanged"
    fi
    cd $MODEL_TARGET_DIR
}

function fn_main()
{
    echo "-----run.sh-----"
    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
        echo "[RUN_OPTION]: $RUN_OPTION"
        shift
    fi
    
    if [[ ! -z "$1" ]];then
        MODEL_PATH=$1
        echo "[MODEL_PATH]: $MODEL_PATH"
        shift
    fi
    
    if [[ ! -z "$1" ]];then
        DEVICE_ID=$1
        echo "[DEVICE_ID]: $DEVICE_ID"
        shift
    fi

    if [[ ! -z "$1" ]];then
        NPU=$1
        echo "[NPU]: $NPU"
        shift
    fi
    
    if [[ $NPU == "310" ]];then
        fn_modeling_prepare_310
    else
        fn_modeling_prepare_910
    fi


    case "${RUN_OPTION}" in
        "--run")
            python $MODEL_TARGET_DIR/run_llama_example.py --model_path $MODEL_PATH --device_id $DEVICE_ID
            ;;
        "--performance")
            python $MODEL_TARGET_DIR/run_llama_performance.py --model_path $MODEL_PATH --device_id $DEVICE_ID
            ;;
        "--precision")
            python $MODEL_TARGET_DIR/run_llama_precision.py --model_path $MODEL_PATH --device_id $DEVICE_ID
            ;;
        "--help")
            echo "run.sh [--run|--performance|--precision] [model script path] [device id] [310 | 910]"
            ;;
        *)
            echo "[error] unknown run option:${RUN_OPTION}"
            echo "please use the script as the following:"
            echo "run.sh [--run|--performance|--precision] [model script path] [device id] [310 | 910]"
            exit -1
            ;;
    esac
}

fn_main "$@"