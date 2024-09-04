# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf llama2 \
    --save-model-type huggingface \
    --model-type GPT \
    --loader mg_mcore \
    --saver mg_mcore \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/Codellama-mcore/ \
    --save-dir ./model_from_hf/Codellama-hf/
