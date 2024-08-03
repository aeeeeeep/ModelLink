# 修改 ascend-toolkit 路径
 source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择你需要的并行策略，--params-dtype bf16 \  结合需要使用
python tools/checkpoint/convert_ckpt.py \
   --model-type GPT \
   --loader llama2_hf \
   --saver megatron \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --load-dir ./model_from_hf/Baichuan-hf/ \
   --save-dir ./model_weights/Baichuan-legacy/ \
   --tokenizer-model ./model_from_hf/Baichuan-hf/tokenizer.model \
   --w-pack True
