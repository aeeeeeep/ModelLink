# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader gemma_hf \
    --saver megatron \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/Gemma-hf/ \
    --save-dir ./model_weights/Gemma-legacy/ \
    --tokenizer-model ./model_from_hf/Gemma-hf/tokenizer.model