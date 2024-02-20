CONVERT_ARGS="
    --no-checking \
    --model-type GPT \
    --loader loader_bloom_mg \
    --saver saver_huggingface \
    --target-tensor-parallel-size 4 \
    --load-dir ./ckpt \
    --save-dir ./save_dir \
    --tokenizer-model None
"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py $CONVERT_ARGS