# 修改 ascend-toolkit 路径
 source /usr/local/Ascend/ascend-toolkit/set_env.sh

 # 权重格式转换
 python tools/checkpoint/convert_ckpt.py \
     --model-type GPT \
     --loader llama2_hf \
     --saver megatron \
     --target-tensor-parallel-size 1 \
     --target-pipeline-parallel-size 1 \
     --add-qkv-bias \
     --load-dir ./model_from_hf/qwen15-hf/ \
     --save-dir ./model_weights/qwen15-legacy/ \
     --tokenizer-model ./model_from_hf/qwen15-hf/tokenizer.json