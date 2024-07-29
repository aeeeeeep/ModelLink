# 修改modelling_qwen.py文件第39行，将：
# SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
# 修改为：
# SUPPORT_FP16 = True

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader qwen_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --load-dir ./model_from_hf/Qwen-hf/ \
    --save-dir ./model_weights/Qwen-legacy/ \
    --tokenizer-model ./model_from_hf/Qwen-hf/qwen.tiktoken \
    --add-qkv-bias