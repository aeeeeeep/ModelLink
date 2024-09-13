# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh


python convert_ckpt.py --model-type GPT \
   --loader megatron \
   --saver megatron \
   --save-model-type save_huggingface_llama \
   --load-dir ./model_weights/Baichuan-legacy/ \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --w-pack True \
   --save-dir ./model_from_hf/Baichuan-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Baichuan-hf/mg2hg/