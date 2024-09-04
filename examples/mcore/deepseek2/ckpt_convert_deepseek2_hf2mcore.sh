# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换，设置需要的并行策略
python tools/checkpoint/convert_ckpt.py \
   --use-mcore-models \
   --moe-grouped-gemm \
   --model-type-hf deepseek2 \
   --model-type GPT \
   --loader hf_mcore \
   --saver mg_mcore \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --target-expert-parallel-size 8 \
   --load-dir ./model_from_hf/deepseek2-hf/ \
   --save-dir ./model_weights/deepseek2-mcore/ \
   --tokenizer-model ./model_from_hf/deepseek2-hf/