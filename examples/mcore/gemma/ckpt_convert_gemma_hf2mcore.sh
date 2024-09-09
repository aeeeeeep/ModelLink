# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf gemma \
    --model-type GPT \
    --loader hf_mcore \
    --saver mg_mcore \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --params-dtype bf16 \
    --load-dir ./model_from_hf/gemma_hf/ \
    --save-dir ./model_weights/gemma_mcore/ \
    --tokenizer-model ./model_from_hf/gemma_hf/tokenizer.model