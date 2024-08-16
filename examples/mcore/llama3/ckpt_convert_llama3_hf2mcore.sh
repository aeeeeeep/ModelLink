# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换，设置需要的并行配置
python tools/checkpoint/convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf llama2 \
    --model-type GPT \
    --loader hf_mcore \
    --saver mg_mcore \
    --params-dtype bf16 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/llama3-hf/ \
    --save-dir ./model_weights/llama3-mcore/ \
    --tokenizer-model ./model_from_hf/llama3-hf/tokenizer.json

