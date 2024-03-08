#!/bin/bash
# coding=utf-8
pid=$$
cur_path=$(cd "$(dirname "$0")" || exit; pwd)

#model options
backend="npu"
model_path=""
model="llama2"
batch_size="1"
seq_len_in="1024"
seq_len_out="1024"
dtype="fp16"
device_list="0"
exe_mode="dynamo"
jit_compile="false"
input_padding="false"
kv_padding="true"

# deploy options
cann_path="/usr/local/Ascend"
distributed_mode="deepspeed"

#dfx options
log_level="3"


args=$@

for arg in ${args}
do
    if [[ "${arg}" =~ "--help" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): LLM INFERENCE TOOLS USAGE:"
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Command Line: bash llm_inference.sh [...]"
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Options:
[Model Options]
  --backend                 String, backend, default is npu, current only support npu
  --model_path              String, Model path
  --model                   String, Model name, default is llama2
  --batch_size              Int, Batch size, default 1
  --seq_len_in              Int, Sequence length of input, default is 1024
  --seq_len_out             Int, Sequence length of output, default is 1024
  --dtype                   String, weight dtype default is fp16
  --device_list             Int/IntList, Specific device index of execution, default is 0
  --exe_mode                String, Execution types, default is dynamo, Support eager and dynamo, dynamo ONLY take effect in pytorch 2.1 scene
  --jit_compile             Bool, Enable binary, default is false
  --input_padding           Bool, enable input_padding or not, default is false
  --kv_padding              Bool, enable KV cache padding or not, default is true

[Deploy Options]
  --cann_path               String, Cann package home, default is /usr/local/Ascend
  --distributed_mode=        String, distributed_mode, support sigle, deepspeed

[Dfx Options]
  --log_level               Int, Set npu slog level, default is 3. Support 0,1,2,3 means DEBUG,INFO,WARNING,ERROR
"
    exit 0
    elif [[ "${arg}" =~ "--backend=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution backend: ${arg#*=}"
        backend=${arg#*=}
    elif [[ "${arg}" =~ "--model_path=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution model path: ${arg#*=}"
        model_path=${arg#*=}
    elif [[ "${arg}" =~ "--model=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution model name: ${arg#*=}"
        model=${arg#*=}
    elif [[ "${arg}" =~ "--batch_size=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution batch size: ${arg#*=}"
        batch_size=${arg#*=}
    elif [[ "${arg}" =~ "--seq_len_in=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution input sequence length: ${arg#*=}"
        seq_len_in=${arg#*=}
    elif [[ "${arg}" =~ "--seq_len_out=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution output sequence length: ${arg#*=}"
        seq_len_out=${arg#*=}
    elif [[ "${arg}" =~ "--dtype=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution dtype: ${arg#*=}"
        dtype=${arg#*=}
    elif [[ "${arg}" =~ "--device_list=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution device list: ${arg#*=}"
        device_list=${arg#*=}
    elif [[ "${arg}" =~ "--input_padding=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution input padding mode: ${arg#*=}"
        input_padding=${arg#*=}
    elif [[ "${arg}" =~ "--kv_padding=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution kv cache padding: ${arg#*=}"
        kv_padding=${arg#*=}
    elif [[ "${arg}" =~ "--exe_mode=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution mode: ${arg#*=}"
        exe_mode=${arg#*=}
    elif [[ "${arg}" =~ "--jit_compile=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution jit compile mode: ${arg#*=}"
        jit_compile=${arg#*=}
    elif [[ "${arg}" =~ "--cann_path=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got execution cann path: ${arg#*=}"
        cann_path=${arg#*=}
    elif [[ "${arg}" =~ "--distributed_mode=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got distributed mode: ${arg#*=}"
        distributed_mode=${arg#*=}
    elif [[ "${arg}" =~ "--log_level=" ]];then
        echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Got log level: ${arg#*=}"
        log_level=${arg#*=}
    fi
done

device_array=(${device_list//,/ })
device_num=${#device_array[@]}

# init environment
if [ ${backend} == "npu" ];then
    source ${cann_path}/latest/bin/setenv.bash
    export ASCEND_GLOBAL_LOG_LEVEL=${log_level}
fi

# write config file
echo "{" > config/llm_inference.json
echo "    \"backend\": \"${backend}\"," >> config/llm_inference.json
echo "    \"model_path\": \"${model_path}\"," >> config/llm_inference.json
echo "    \"model\": \"${model}\"," >> config/llm_inference.json
echo "    \"batch_size\": ${batch_size}," >> config/llm_inference.json
echo "    \"seq_len_in\": ${seq_len_in}," >> config/llm_inference.json
echo "    \"seq_len_out\": ${seq_len_out}," >> config/llm_inference.json
echo "    \"dtype\": \"${dtype}\"," >> config/llm_inference.json
echo "    \"device_list\": \"${device_list}\"," >> config/llm_inference.json
echo "    \"input_padding\": ${input_padding}," >> config/llm_inference.json
echo "    \"kv_padding\": ${kv_padding}," >> config/llm_inference.json
echo "    \"exe_mode\": \"${exe_mode}\"," >> config/llm_inference.json
echo "    \"jit_compile\": ${jit_compile}," >> config/llm_inference.json
echo "    \"distributed_mode\": \"${distributed_mode}\"," >> config/llm_inference.json
echo "    \"cann_path\": \"${cann_path}\"," >> config/llm_inference.json
echo "    \"log_level\": ${log_level}" >> config/llm_inference.json
echo "}" >> config/llm_inference.json

while true
do
    min_port=50000
    threshold=10000
    port=$(($((RANDOM+10000000000))%${threshold}+${min_port}))
    port_occupy=`lsof -i:${port}`
    if [ "A${port_occupy}" == "A" ];then
        break
    fi
done

if [ ${distributed_mode} == "single" ];then
    if [ ${backend} == "npu" ];then
        export ASCEND_DEVICE_ID=${device_list}
    fi
    export LOCAL_RANK=${device_list}
    echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Start to execute: python3 llm_inference.py in sigle mode"
    python3 llm_inference.py
elif [ ${distributed_mode} == "deepspeed" ];then
    echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - INFO - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): Start to execute: deepspeed --num_gpus=xxx llm_inference.py"
    deepspeed --num_gpus=${device_num} --master_port=${port} llm_inference.py
else
   echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - ERROR - [LLM](${BASH_SOURCE[0]##*/}:$LINENO): distributed_mode=${distributed_mode} not supported"
fi
