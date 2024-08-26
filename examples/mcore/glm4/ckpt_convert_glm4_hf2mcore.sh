# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python tools/checkpoint/convert_ckpt.py \
   --model-type-hf chatglm3 \
   --model-type GPT \
   --loader hf_mcore \
   --saver mg_mcore \
   --target-tensor-parallel-size 2 \
   --target-pipeline-parallel-size 2 \
   --load-dir ./model_from_hf/glm4-hf \
   --save-dir ./model_weights/glm4-mcore \
   --tokenizer-model ./model_from_hf/glm4-hf/tokenizer.json \
   --add-qkv-bias \
   --use-mcore-models \
   --params-dtype bf16
