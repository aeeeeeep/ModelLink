# 设置需要的权重转换参数
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 4 \
       --target-expert-parallel-size 1 \
       --add-qkv-bias \
       --load-dir ./model_from_hf/qwen2_moe_hf/ \
       --save-dir ./model_weights/qwen2_moe_mcore/ \
       --tokenizer-model ./model_from_hf/qwen2_moe_hf/tokenizer.json \
       --model-type-hf qwen2-moe \
       --moe-grouped-gemm \
       --num-layer-list 6,7,7,8 \
       --params-dtype bf16 # --target-tensor-parallel-size  --target-pipeline-parallel-size --target-expert-parallel-size
                           # 这三个参数需要根据实际需求进行修改， 开启动态流水线并行时，需要根据脚本的参数修改--num-layer-list的值