# 修改modelling_qwen.py文件第39行，将：
# SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
# 修改为：
# SUPPORT_FP16 = True

# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_qwen \
    --load-dir ./model_weights/Qwen-legacy/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --save-dir ./model_from_hf/Qwen-hf/     # 需要填入原始HF模型路径，新权重会存于./model_from_hf/Qwen-hf/mg2hg/