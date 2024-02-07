#!/bin/bash
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=110
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=0
export ATB_CONTEXT_WORKSPACE_RING=1
export ATB_USE_TILING_COPY_STREAM=0
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export LCCL_ENABLE_FALLBACK=1
export MAX_SEQ_LENGTH=2048
export LONG_SEQ_ENABLE=0
export BIND_CPU=0

script_dir=$(cd $(dirname $0); pwd)
transformers_package_path=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
cp $script_dir/modeling_llama_ascend.py $transformers_package_path/models/llama/modeling_llama.py

python3 sdk_test.py --task run