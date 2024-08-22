# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader hf_mcore \
    --saver mg_mcore \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/chatglm3-hf/ \
    --save-dir ./model_weights/chatglm3-mcore/ \
    --tokenizer-model ./model_from_hf/chatglm3-hf/tokenizer.model \
    --use-mcore-models \
    --add-qkv-bias \
    --model-type-hf chatglm3
