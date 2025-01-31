# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 8 \
    --load-dir ./model_from_hf/Llama-hf/ \
    --save-dir ./model_weights/Llama-legacy/ \
    --tokenizer-model ./model_from_hf/Llama-hf/tokenizer.model