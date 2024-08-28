# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

python ./preprocess_data.py \
   --input ./dataset/wikipedia-cn-20230720-filtered.json \
   --tokenizer-name-or-path ./model_from_hf/gemma_hf/ \
   --output-prefix ./dataset/wikipedia_cn \
   --json-key completion \
   --workers 4 \
   --log-interval 1000  \
   --tokenizer-type PretrainedFromHF