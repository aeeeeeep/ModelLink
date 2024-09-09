# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --loader hf_mcore \
    --saver mg_mcore \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 8 \
    --load-dir ./model_from_hf/qwen1.5_110B_hf/ \
    --save-dir ./model_weights/qwen1.5_110B_mcore/ \
    --tokenizer-model ./model_from_hf/qwen1.5_110B_hf/tokenizer.model \
    --use-mcore-models \
    --model-type-hf llama2 \
    --add-qkv-bias \
    --params-dtype bf16 \
    --num-layer-list 7,8,9,10,11,11,12,12
