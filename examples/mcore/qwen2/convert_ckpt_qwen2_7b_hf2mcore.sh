# 修改 ascend-toolkit 路径
 source /usr/local/Ascend/ascend-toolkit/set_env.sh

 # 权重格式转换
 python tools/checkpoint/convert_ckpt_qwen2.py \
     --use-mcore-models \
     --model-type GPT \
     --loader qwen2_hf_mcore \
     --saver qwen2_mg_mcore \
     --target-tensor-parallel-size 1 \
     --target-pipeline-parallel-size 2 \
     --params-dtype bf16 \
     --add-qkv-bias \
     --load-dir ./model_from_hf/qwen2-7B-hf/  \
     --save-dir ./model_weights/qwen2-7B-mcore-t1p2  \
     --tokenizer-model ./model_from_hf/qwen2-7B-hf/tokenizer.json

