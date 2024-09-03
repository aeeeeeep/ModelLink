# Alpaca数据集下载链接: https://huggingface.co/datasets/tatsu-lab/alpaca
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path ./model_from_hf/Llama2-hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --overwrite-cache \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2 \
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传