### 利用LlamaTokenizer来训练Llama模型教程

1. 下载Llama的tokenizer模型文件：
    - 下载路径：https://onebox.huawei.com/p/8c64558b33abb4904dc46e70a39252bf
    - tokenizer.model: tokenizer的模型文件
    - tokenizer_config.json: tokenizer的配置文件，用于加载huggingFace中的llamaTokenizer
    - special_tokens_map.json: 特殊字符文件

2. 修改训练脚本文件中的--vocab_file的参数，修改为如上模型文件解压后所在的路径。
    - 如：文件解压的路径为/data/llama_tokenizer/，则--vocab_file=/data/llama_tokenizer/

3. 去除训练脚本文件中的--merge-file参数。

4. `pretrain_llama.py`文件中的 args_defaults的tokenizer_type修改为：'LLamaTokenizer'
