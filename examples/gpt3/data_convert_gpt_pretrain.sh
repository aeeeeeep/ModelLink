# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

# 下载 vocab file 和 merge table
# cd vocab_file
# wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
# wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
# cd ..

# 处理成训练数据
python ./preprocess_data.py \
    --input ./dataset/ \
    --output-prefix ./dataset/gpt_text_sentence \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file ./vocab_file/gpt2-vocab.json \
    --merge-file ./vocab_file/gpt2-merges.txt \
    --append-eod \
    --workers 4 \
    --log-interval 1000