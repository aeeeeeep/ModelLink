# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的权重转换参数
python tools/checkpoint/convert_ckpt.py \
   --model-type GPT \
   --loader mixtral_hf \
   --saver mixtral \
   --load-dir ./model_from_hf/Mixtral-hf/ \
   --save-dir ./model_weights/Mixtral-legacy/ \
   --tokenizer-model ./model_from_hf/Mixtral-hf/tokenizer.model \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 4 \
   --target-expert-parallel-size 1