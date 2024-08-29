# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --model-type-hf gemma \
    --model-type GPT \
    --loader mg_mcore \
    --saver mg_mcore \
    --save-model-type huggingface \
    --load-dir ./model_weights/gemma-legacy/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/gemma-hf/     # 需要填入原始HF模型路径，新权重会存于./model_from_hf/Gemma-hf/mg2hg/