source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/Aquila-legacy/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/Aquila-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Aquila-hf/mg2hg/