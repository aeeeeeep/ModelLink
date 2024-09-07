# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --moe-grouped-gemm \
   --model-type-hf deepseek2 \
   --save-model-type huggingface \
   --model-type GPT \
   --loader mg_mcore \
   --saver mg_mcore \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --target-expert-parallel-size 1 \
   --load-dir ./model_weights/deepseek-mcore/ \
   --save-dir ./model_from_hf/deepseek2-hf/