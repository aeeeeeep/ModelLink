# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --loader mg_mcore \
    --saver mg_mcore \
    --save-model-type huggingface \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --model-type-hf bloom \
    --add-qkv-bias \
    --add-dense-bias \
    --load-dir ./model_weights/Bloom-legacy/ \
    --save-dir ./model_from_hf/Bloom-hf/   # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Bloom-hf/mg2hg/
#    --params-dtype bf16 \  结合需要使用