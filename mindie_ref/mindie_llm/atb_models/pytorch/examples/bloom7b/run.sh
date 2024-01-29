#!/bin/bash
export HCCL_BUFFSIZE=110
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_USE_TILING_COPY_STREAM=1
export ATB_CONTEXT_WORKSPACE_RING=1
# export ATB_LAYER_INTERNAL_TENSOR_REUSE=1

# 日志开关
# export ATB_LOG_LEVEL=INFO
# export ATB_LOG_TO_STDOUT=1
# export ATB_LOG_TO_FILE=1

# export ASDOPS_LOG_LEVEL=INFO
# export ASDOPS_LOG_TO_STDOUT=1
# export ASDOPS_LOG_TO_FILE=1

# 原始权重的路径
input_dir="./bloom"
# ln -sf /your/bloom/model/path/* ./
# export FLOAT_LAYERS="7"
# a number like "7", or multi number like "6,7,8", or "all"

output_dir=$FLOAT_MODEL_PATH

SCRIPT_DIR=$(cd $(dirname $0); pwd)

world_size_=2

options=$(getopt -o p:d:c:e --long patch:,device:,cut,ceval -- "$@")
eval set -- "$options"

while true; do
  case $1 in 
  	-p | --patch) shift; patch=$1 ; shift ;;
    -d | --device) shift; device=$1 ; shift ;;
    -c | --cut) cut=true; shift ;;
    -e | --ceval) ceval=true; shift ;;
    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done

if [ -z "$patch" ];then
    SCRIPT_PATH=$SCRIPT_DIR/patches/models/modeling_bloom_model_quant.py
else
    SCRIPT_PATH=$(cd $(dirname $patch); pwd)/$(basename $patch)
fi
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/models/bloom

if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_bloom_origin.py" ];then
    rm -rf $TRANSFORMER_PACKAGE_PATH/modeling_bloom_origin.py
fi

if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_bloom.py" ];then
    mv $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py.bak
fi

if [ ! -f $SCRIPT_PATH ];then
    echo "cannot find the file to be tested"
    exit 1
fi

cp $SCRIPT_DIR/modeling_bloom.py $TRANSFORMER_PACKAGE_PATH/modeling_bloom_origin.py
cp $SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py

RUN_OPTION="--run"
if [ ! -z $cut ];then
    RUN_OPTION="--cut"
fi
if [ ! -z $ceval ];then
    RUN_OPTION="--ceval"
fi
cd $SCRIPT_DIR
echo $RUN_OPTION

devices=`echo "$device" | sed 's/,/ /g'`
echo "use npu $devices"
device_length=`expr length "$device"`

if [ $device_length -gt 1 ]
then
    # 双芯量化权重的路径
    export QUANT_MODEL_PATH='./bloom_quant_cut'
    # 双芯浮点权重的路径
    export FLOAT_MODEL_PATH="./bloom_cut"
else
    # 量化权重的路径
    export QUANT_MODEL_PATH='./bloom_quant'
    # 浮点权重的路径
    export FLOAT_MODEL_PATH="./bloom"
fi

case "${RUN_OPTION}" in
    "--run")
        if [ $device_length -gt 1 ]
        then
            export ASCEND_RT_VISIBLE_DEVICES=$device
            torchrun --nproc_per_node 2 --master_port 39682 run_bloom_npu.py --load_path $FLOAT_MODEL_PATH --device $devices
        else
            python3 $SCRIPT_DIR/run_bloom_npu.py --load_path $FLOAT_MODEL_PATH --device $devices
        fi
        ;;
    "--cut")
        echo "cutting the weight......"
        cp $SCRIPT_DIR/modeling_bloom_parallel.py $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py
        python3 $SCRIPT_DIR/cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_
        ;;
    "--ceval")
        echo "testing ceval....."
        export PYTHONPATH=$(dirname $SCRIPT_DIR):$PYTHONPATH
        export ASCEND_RT_VISIBLE_DEVICES=$device
        torchrun --nproc_per_node 2 --master_port 39782 $SCRIPT_DIR/ceval_test.py --load_path $FLOAT_MODEL_PATH --device $devices
        ;;
    *)
        echo "unknown build type:${RUN_OPTION}"
        echo "run.sh [model script path] [--run|--zhipu]"
        ;;
esac

rm -f $TRANSFORMER_PACKAGE_PATH/modeling_bloom_origin.py
if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_bloom.py.bak" ];then
    mv $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py.bak $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py
fi