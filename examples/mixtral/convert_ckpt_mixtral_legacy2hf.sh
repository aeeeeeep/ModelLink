# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
   --model-type GPT \
   --loader mixtral_mg \
   --saver mixtral \
   --save-model-type huggingface \
   --load-dir ./model_weights/Mixtral-legacy/ \
   --save-dir ./model_from_hf/Mixtral-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Mixtral-hf/mg2hg/
