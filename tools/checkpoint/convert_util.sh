CONVERT_ARGS="
    --no-checking \
    --model-type GPT \
    --loader loader_llama2_hf \
    --saver saver_megatron \
    --target-tensor-parallel-size 4 \
    --load-dir ./ckpt \
    --save-dir ./save_dir \
    --tokenizer-model None
"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py $CONVERT_ARGS