export CUDA_DEVICE_MAX_CONNECTIONS=1

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的并行配置
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader hf_mcore \
    --saver mg_mcore \
    --params-dtype bf16 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --load-dir ./model_from_hf/Mixtral-hf/ \
    --save-dir ./model_weights/Mixtral-mcore/ \
    --tokenizer-model ./model_from_hf/Mixtral-hf/tokenizer.model \
    --use-mcore-models \
    --model-type-hf mixtral
