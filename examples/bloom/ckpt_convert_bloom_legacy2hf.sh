# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/Bloom-legacy/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --embed-layernorm \
    --save-dir ./model_from_hf/Bloom-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Bloom-hf/mg2hg/

#    --params-dtype bf16 \  结合需要使用