# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader hf_mcore \
    --saver mg_mcore \
    --model-type-hf chatglm3 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/Chatglm3-hf/ \
    --save-dir ./model_weights/Chatglm3-legacy/ \
    --tokenizer-model ./model_from_hf/Chatglm3-hf/tokenizer.model \
    --add-qkv-bias