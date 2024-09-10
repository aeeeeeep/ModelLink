# ModelLink 开发使用指南

<p align="left">
</p>

## 目录


- [环境安装](#环境安装)
- [权重下载及转换](#权重下载及转换)
- [数据集准备及处理](#数据集准备及处理)
- [大模型分布式预训练](#大模型分布式预训练)
- [大模型分布式推理](#大模型分布式推理)
- [大模型分布式评估](#大模型分布式评估)
- [社区BUG列表](#社区BUG列表)

---

## 环境安装

【模型开发时推荐使用配套的环境版本】

|    软件     | [版本](https://www.hiascend.com/zh/) |
|:---------:|:----------------------------------:|
|  Python   |                3.8                 |
|  Driver   |         在研版本          |
| Firmware  |         在研版本          |
|   CANN    |             在研版本             |
|   Torch   |            2.1.0、2.2.0             |
| Torch_npu |           在研版本           |


#### 1. 克隆仓库到本地服务器

```shell
    git clone https://gitee.com/ascend/ModelLink.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../ModelLink/
    cd ..
    cd ModelLink
    mkdir logs
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
```
#### 2. 环境搭建

```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.6.0
    git checkout e6ea2117 
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -r requirements.txt 
```

---

## 权重下载及转换

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

预训练权重链接在 [首页README.md](../README.md) `配套版本与支持模型`章节列表的`参数`列链接中可以获取

更多社区资源可以在`模型`列链接中获取，如`Chat/Instruct`权重等

权重可以基于网页直接下载，也可以基于命令行下载，保存到ModelLink/model_from_hf目录，比如：


```shell
#!/bin/bash
mkdir ./model_from_hf/llama-2-7b-hf/
cd ./model_from_hf/llama-2-7b-hf/
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/generation_config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00002-of-00002.bin
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/special_tokens_map.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
cd ../../
```

#### 2. 权重转换

##### 2.1 Huggingface权重转换到Megatron

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --num-layer-list 8,8,8,8 \
    --model-type-hf llama2 \
    --load-dir ./model_from_hf/llama-2-7b-hf/ \
    --save-dir ./model_weights/llama-2-7b-legacy/ \
    --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model
```

【--target-tensor-parallel-size】

指明需要切分的TP数量，默认为1

【--target-pipeline-parallel-size】

指明需要切分的PP数量，默认为1

【--num-layer-list】

可选参数，支持动态PP划分，通过列表指定每个PP Stage的层数

【--num-layers-per-virtual-pipeline-stage】

可选参数，支持VPP划分，指定VPP的每个Stage层数，默认为None

注意：VPP和动态PP划分只能二选一

【--use-mcore-models】

设置是否转换为Megatron-Mcore权重，若不指定，则默认转换为Megatron-Legacy权重

【--model-type-hf】

huggingface模型类别，默认为llama2，目前支持的模型见 [model_cfg.json](https://gitee.com/ascend/ModelLink/blob/master/modellink/tasks/checkpoint/model_cfg.json)

【--tokenizer-model】

需要指明到具体的分词器模型文件，如 tokenizer.model、tokenizer.json、qwen.tiktoken、None等，具体取决于huggingface中词表文件的格式形式

【--params-dtype】

指定权重转换后的权重精度模式，默认为fp16，如果源格式文件为bf16，则需要对应设置为bf16，影响推理或评估结果

【启动脚本】

ModelLink Huggingface到Megatron-Legacy权重转换脚本命名风格及启动方法为：
```shell
# 命名及启动：bash examples/model_name/ckpt_convert_xxx_hf2legacy.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/llama2/ckpt_convert_llama2_hf2legacy.sh
```

ModelLink Huggingface到Megatron-Mcore权重转换脚本命名风格及启动方法为：
```shell
# 命名及启动：bash examples/model_name/ckpt_convert_xxx_hf2mcore.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/llama2/ckpt_convert_llama2_hf2mcore.sh
```

##### 2.2 Megatron权重转换到Huggingface

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --model-type-hf llama2 \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hf/
```
参数意义参考2.1

【启动脚本】

ModelLink Megatron-Legacy到Huggingface的权重转换脚本命名风格及启动方法为：
```shell
# 命名及启动：bash examples/model_name/ckpt_convert_xxx_legacy2hf.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/llama2/ckpt_convert_llama2_legacy2hf.sh
```

ModelLink Megatron-Mcore到Huggingface的权重转换脚本命名风格及启动方法为：
```shell
# 命名及启动：bash examples/model_name/ckpt_convert_xxx_mcore2hf.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/llama2/ckpt_convert_llama2_mcore2hf.sh
```

##### 2.3 Megatron权重互转

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# legacy转legacy
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --save-dir ./model_weights/llama-2-7b-legacy_tp2pp2/

# legacy转mcore
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --load-from-legacy \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --save-dir ./model_weights/llama-2-7b-mcore_tp2pp2/

# mcore转mocre
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --save-dir ./model_weights/llama-2-7b-mcore_tp2pp2/

# mcore转legacy
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --save-to-legacy \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --save-dir ./model_weights/llama-2-7b-legacy_tp2pp2/
```
【load-from-legacy】 

legacy转mcore时设置此参数以指定导入权重格式为legacy

【save-to-legacy】 

mcore转legacy时设置此参数以指定保存权重格式为legacy

其余参数意义参考2.1

注：上述权重legacy和mcore互转为高阶功能，modellink基于llama2提供基础能力，并进行版本迭代看护，其余模型的支持需要用户自行修改支持

##### 2.4 lora权重与base权重合并

在上述权重转换命令中，加入如下参数可以将训练的 lora 权重与base进行融合。

```bash
--lora-load ${CHECKPOINT_LORA}  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
```

【合并后转换为Megatron-Legacy权重】

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_weights/llama-2-7b-lora2legacy
```

转换脚本命名风格及启动方法为：
```shell
bash examples/llama2/ckpt_convert_llama2_legacy2legacy_lora.sh
```

【合并后转换为Huggingface权重】

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hg/
```

转换脚本命名风格及启动方法为：
```shell
bash examples/llama2/ckpt_convert_llama2_legacy2hf_lora.sh
```

**注意：** lora参数值需与lora微调时的参数保持一致


---

## 数据集准备及处理

#### 1. 数据集下载

从Huggingface等网站下载开源数据集，保存到ModelLink/dataset/ 目录

常用的预训练数据集有：
- [Enwiki数据集](https://huggingface.co/datasets/lsb/enwiki20230101)
- [C4数据集](https://huggingface.co/datasets/allenai/c4)
- [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

常用的对话指令微调数据集有：

- [单轮对话：Alpaca数据集](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [多轮对话：ShareGPT数据集](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data)
- [多轮对话：AlpacaHistroy数据集](https://huggingface.co/datasets/kimnt93/oaast-selected)

数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```shell
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/lsb/enwiki20230101/resolve/main/data/train-00000-of-00042-d964455e17e96d5a.parquet
cd ..
```

#### 2. 数据集处理

##### 2.1 预训练数据集处理方法

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-hf \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ./dataset/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000  
```

【--input】

可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持 .parquet \ .csv \ .json \ .jsonl \ .txt \ .arrow 格式， 同一个文件夹下的数据格式需要保持一致 

【--handler-name】

当前预训练默认使用 `GeneralPretrainHandler`，支持的是预训练数据风格，提取数据的`text`列，格式如下：

```shell
[
  {"text": "document"},
  {"other keys": "optional content"}
]
```

用户可结合具体数据处理需求添加新的Handler进行数据处理 

【--json-keys】

从文件中提取的列名列表，默认为 `text`，可以为 `text`, `input`, `title` 等多个输入，结合具体需求及数据集内容使用，如：
```shell
--json-keys text input output \
```

【--n-subs】

数据预处理并行加速参数。当需要预处理的数据集比较大时，可以通过并行处理进行加速，方法为设置参数`--n-subs`，通过该参数设置并行处理数量。在数据预处理过程会将原始数据集切分为`n_sub`个子集，对子集进行并行处理，然后合并，从而实现加速。建议预处理数据集超过GB级别时加上该参数。


ModelLink预训练数据集处理脚本命名风格及启动方法为：
```shell
# Legacy
# 命名及启动：examples/model_name/data_convert_xxx_pretrain.sh
bash examples/llama2/data_convert_llama2_pretrain.sh

# Mcore
# 命名及启动：examples/mcore/model_name/data_convert_xxx_pretrain.sh
bash examples/mcore/llama2/data_convert_llama2_pretrain.sh
```

预训练数据集处理结果如下：
```shell
./dataset/enwiki_text_document.bin
./dataset/enwiki_text_document.idx
```

预训练时，数据集路径输入 ./dataset/enwiki_text_document 即可

##### 2.2 微调数据集处理方法
###### 2.2.1 Alpaca风格数据集处理方法
Alpaca风格微调数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：
```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

在指令监督微调时，instruction 列对应的内容会与 input 列对应的内容拼接后作为人类指令，即人类指令为 instruction\ninput。而 output 列对应的内容为模型回答。如果指定了history，则会将历史对话内容也加入进来。如果指定system 列，则对应的内容将被作为系统提示词。

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2  # <-- 需要填入模型模板
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传
```

【--input】

可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持 .parquet \ .csv \ .json \ .jsonl \ .txt \ .arrow 格式， 同一个文件夹下的数据格式需要保持一致 

【--map-keys】

`--map-keys`参数用于配置字段映射来使用数据集。

Alpaca风格示例：
```
[
{
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
    ["第一轮指令（选填）", "第一轮回答（选填）"],
    ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
}
]
```

对于上面格式的数据，`--map-keys`参数完整应为

`'{"prompt":"instruction","query":"input","response":"output","system":"system","history":"history"}'`

其中参数的key值`"prompt"、"query"、"response"、"system"、"history"`代表数据集列映射后的属性，在代码中是固定的，不应改变，value值`"instruction"、"input"、"output"、"system"、"history"`对应数据集的列名。

考虑到alpaca数据集大部分都是`["instruction", "input", "output"]`型格式，因此我们为key值`["prompt", "query", "response"]`设置了默认值。因此上面格式`--map-keys`参数可简略为`'{"system": "system","history": "history"}'`

若数据集中无`system`与`history`列，则`--map-keys`可省略。


【--prompt-type】


用于指定模型模板，能够让base模型微调后能具备更好的对话能力。

目前支持的模板有：

`['empty', 'default', 'chatglm3_system', 'chatml', 'qwen', llama2]`

【--handler-name】

微调数据预处理Alpaca风格数据集时，应指定为`AlpacaStyleInstructionHandler`，根据`--map-keys`参数提取对应数据的列。

**示例1：**
```
    --map-keys '{"prompt":"notice","query":"question","response":"answer","system":"system_test","history":"histories"}'
```
则会提取数据集里的`"notice"、"question"、"answer"、"system_test"、"histories"`列

**示例2：**
```
    --map-keys '{"history":"histories"}'
```
则会提取数据集里的`"instruction"、"input"、"output"、"histories"`列，其中`"instruction"、"input"、"output"`列作为默认值隐式存在。


###### 2.2.2 Sharegpt风格数据集处理方法

相比 alpaca 格式的数据集，sharegpt 格式支持更多的角色种类，例如 `human、gpt、observation、function`等等。它们构成一个对象列表呈现在`conversations`列中。

Sharegpt风格示例：
```
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "function_call",
        "value": "工具参数"
      },
      {
        "from": "observation",
        "value": "工具结果"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
]
```
Sharegpt风格微调数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：
```shell
cd dataset/
wget https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data/resolve/main/sharegpt_formatted_data-evol-gpt4.jsonl
cd ..
```
Sharegpt格式数据预处理脚本：
```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/sharegpt_formatted_data-evol-gpt4.jsonl \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/sharegpt \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type llama2  # <-- 需要填入模型模板
    # --map-keys '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}' # 默认值，可不传
```

【--map-keys】

`--map-keys`参数用于配置字段映射来使用数据集。默认值为

`'{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}'`

其中key值`"messages"、"tags"`代表数据集列映射后的属性，在代码中是固定的，不应改变。value值中`"conversations"`对应数据集的列名、`"from"`对应角色标志、`"human"、"gpt"、"system"、"observation"、"function_call"`对应角色种类、`"value"`对应具体内容标志。


以OpenAI格式为例，OpenAI 格式是 sharegpt 格式的一种特殊情况，其中第一条消息可能是系统提示词。

OpenAI格式示例：

```
[
  {
    "messages": [
      {
        "role": "system",
        "content": "系统提示词（选填）"
      },
      {
        "role": "user",
        "content": "人类指令"
      },
      {
        "role": "assistant",
        "content": "模型回答"
      }
    ]
  }
]
```
OpenAI格式数据预处理脚本：

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/xxx.json \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/openai \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type llama2 \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
```

【--handler-name】

微调数据预处理Sharegpt风格数据集时，应指定为`SharegptStyleInstructionHandler`，并根据`--map-keys`参数提取对应数据的列。

**示例1：**
```
    --map-keys '{"messages":"chat"}'
```
则会提取数据集里的`"chat"`列，其中`"tags"`属性包含角色格式和内容格式，做为默认值隐式存在，角色格式可以为：`"from": "human"、"from": "gpt"、"from": "observation"、"from": "function_call"`，内容格式为`"value": "具体内容"`

**示例2：**
```
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant"}}'
```
则会提取数据集里的`"messages"`列，其中角色格式可以为：`"role": "user"、"role": "assistant"`，内容格式为`"content": "具体内容"`


ModelLink微调数据集处理脚本命名风格及启动方法为：
```shell
# Legacy
# 命名及启动：examples/model_name/data_convert_xxx_instruction.sh
bash examples/llama2/data_convert_llama2_instruction.sh
```

指令微调数据集处理结果如下：
```shell
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```

微调时，数据集路径输入 ./finetune_dataset/alpaca 即可


#### 3. 数据集合并

若要对预处理好的多个数据集进行合并，须将待合并数据集放在一个单独文件夹里面，然后按如下调用命令：

预训练：

```shell
python ./merge_datasets.py \
    --input ./process_data/enwiki_subsets \
    --output-prefix ./process_data/merge_enwiki
    # --keys text_document # 默认值，可不传
```

微调：

```shell
python ./merge_datasets.py \
    --input ./process_data/alpaca_tune_subsets \
    --output-prefix ./process_data/merge_tune_alpaca
    --keys packed_attention_mask_document packed_input_ids_document packed_labels_document
```

【--input】

参数值为待合并数据集所在文件夹路径，里面应包含每个数据处理之后的.bin和.idx文件，例如有两个预训练数据集，则应包含四个文件（若为微调，则两个数据集应包含12个文件）：

data1_xxx_text_document.idx, data1_xxx_text_document.bin, data2_xxx_text_document.idx, data2_xxx_text_document.bin

【--output-prefix】

参数值为合并之后数据集保存路径前缀

【--keys】

- 参数值为文件分类标识符列表，文件夹中文件名匹配到含有相同key的文件会被合并
- 合并之后生成的数据集文件命名规则为：`output_prefix_key`。默认值为预训练场景的数据集文件后缀，即[`text_document`]。微调场景须手动指定keys，如上示例。


---


## 大模型分布式预训练

#### 1. 准备工作
配置脚本前需要完成前置准备工作，包括：**环境安装**、**数据集准备及处理**、**Huggingface权重转换**，详情可查看对应章节

#### 2. 配置预训练参数

legacy分支的预训练脚本保存在 example 中各模型文件夹下：pretrain_xxx_xx.sh

mcore分支的预训练脚本保存在 example/mcore 中各模型文件夹下：pretrain_xxx_xx.sh

需根据实际情况修改路径和参数值：

**示例：** 

examples/llama2/pretrain_llama2_7b_ptd.sh      *(legacy分支)*
examples/mcore/llama2/pretrain_llama2_7b_ptd.sh *(mcore分支)*

路径配置：包括**权重保存路径**、**权重加载路径**、**词表路径**、**数据集路径**
 ```shell
    # 根据实际情况配置权重保存、权重加载、词表、数据集路径
    CKPT_SAVE_DIR="./ckpt/llama-2-7b"  #权重保存路径
    CKPT_LOAD_DIR="./model_weights/llama-2-7b-legacy/"  #权重加载路径
    TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf/tokenizer.model"  #词表路径
    DATA_PATH="./dataset/enwiki_text_document"  #数据集路径
```
【--tokenizer-type】 

参数值为PretrainedFromHF时， 词表路径仅需要填到模型文件夹即可，不需要到tokenizer.model文件

【--data-path】 

支持多数据集训练，参数格式如下

```shell 
    --data-path dataset1-weight dataset1-path dataset2-weight dataset2-path
```
**示例：**
```shell 
    --data-path 0.5 "./dataset/enwiki_text_document1" 0.5 "./dataset/enwiki_text_document2"
```

【单机运行】 
```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=6000
    NNODES=1  
    NODE_RANK=0  
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```
【多机运行】 
```shell
    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8  #每个节点的卡数
    MASTER_ADDR="your master node IP"  #都需要修改为主节点的IP地址（不能为localhost）
    MASTER_PORT=6000
    NNODES=2  #集群里的节点数，以实际情况填写,
    NODE_RANK="current node id"  #当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```


#### 3. 启动预训练

【legacy分支】 
```shell
    bash example/模型文件夹/pretrain_xxx_xxx.sh
```
**示例：** *(以llama2-7B为例)*
```shell
    bash examples/llama2/pretrain_llama2_7b_ptd.sh
```

【mcore分支】 
```shell
    bash example/mcore/模型文件夹/pretrain_xxx_xxx.sh
```

**示例：** 
```shell
    bash examples/mcore/llama2/pretrain_llama2_7b_ptd.sh
```
**注意**：
- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据


---

## 大模型分布式推理

#### 1. Generate：流式推理

ModelLink 流式推理脚本命名风格及启动方法为：
```shell
# Legacy
# 命名及启动：examples/model_name/generate_xxx.sh
bash examples/llama2/generate_llama2_7b_ptd.sh

# Mcore
# 命名及启动：examples/mcore/model_name/generate_xxx.sh
bash examples/mcore/llama2/generate_llama2_7b_ptd.sh
```

```shell
# 按实际情况修改启动脚本中模型权重路径和分词器路径
CHECKPOINT="./model_weights/llama-2-7b-legacy"
TOKENIZER_PATH="./model_from_hf/llama-2-hf/"

# 启动任务
bash examples/llama2/generate_llama2_7b_ptd.sh
```


---

## 大模型分布式评估

ModelLink 基准评估脚本命名风格及启动方法为：
```shell
# Legacy
# 命名及启动：examples/model_name/evaluate_xxx.sh
bash examples/llama2/evaluate_llama2_7b_ptd.sh

# Mcore
# 命名及启动：examples/mcore/model_name/evaluate_xxx.sh
bash examples/mcore/llama2/evaluate_llama2_7b_ptd.sh
```

```shell
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/llama-2-hf/"  #词表路径
CHECKPOINT="./model_weights/llama-2-7b-legacy"  #权重路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"

# 启动评估脚本
bash examples/llama2/evaluate_llama2_7B_ptd.sh
```
【lora权重评估】

使用lora权重的评估脚本命名风格及启动方法为：

```shell
bash examples/llama2/evaluate_llama2_7B_lora_ptd.sh
```


ModelLink已支持模型评估分数如下：


| 模型            | 任务                                                                        | ModelLink | 社区值                                                                   | 模型           | 任务                                                                     | ModelLink | 社区值                                                                 |
|---------------|---------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------|--------------|------------------------------------------------------------------------|-----------|---------------------------------------------------------------------|
| Aquila-7B     | [BoolQ](https://github.com/google-research-datasets/boolean-questions)    | 77.3%     | --                                                                    | Aquila2-7B   | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 77.8%     | 77.6%                                                               |
| Aquila2-34B   | [BoolQ](https://github.com/google-research-datasets/boolean-questions)    | 88.0%     | 87.0%                                                                 | Baichuan-7B  | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 69.0%     | [67.0%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)        |
| Baichuan-13B  | [BoolQ](https://github.com/google-research-datasets/boolean-questions)    | 74.7%     | [73.6%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)          | Baichuan2-7B | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 70.0%     | [63.2%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)        |
| Baichuan2-13B | [BoolQ](https://github.com/google-research-datasets/boolean-questions)    | 78.0%     | [67.0%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)          | Bloom-7B     | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                   | 25.1%     | 25.4%                                                               |
| Bloom-176B    | [BoolQ](https://github.com/google-research-datasets/boolean-questions)    | 64.5%     | --                                                                    | ChatGLM3-6B  | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                   | 61.5%     | 61.4%                                                               |
| GLM4-9B       | [MMLU](https://paperswithcode.com/dataset/mmlu)              | 74.5%     | [74.7%](https://huggingface.co/THUDM/glm-4-9b) | --           | --                                                                     | --        | --                                                                  |
| CodeLLaMA-34B | <a href="https://huggingface.co/datasets/openai_humaneval">Human Eval</a> | 48.78%    | [48.8%](https://paperswithcode.com/sota/code-generation-on-humaneval) | Gemma-2B     | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                    | 39.6%     | 39.7%                                                               |
| Gemma-7B      | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                  | 52.2%     | 52.2%                                                                 | InternLM-7B  | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                    | 48.7%     | [51.0](https://huggingface.co/internlm/internlm-7b)                 |
| Gemma2-9B     | [MMLU](https://paperswithcode.com/dataset/mmlu)              | 70.7%     | [71.3%](https://huggingface.co/google/gemma-2-9b)                     | Gemma2-27B   | [MMLU](https://paperswithcode.com/dataset/mmlu)              | 75.5%     | [75.2%](https://huggingface.co/google/gemma-2-27b)                  |
| LLaMA-7B      | [BoolQ](https://github.com/google-research-datasets/boolean-questions)    | 74.6%     | [75.4](https://hub.opencompass.org.cn/dataset-detail/BoolQ)           | LLaMA-13B    | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 79.6%     | [78.7](https://hub.opencompass.org.cn/dataset-detail/BoolQ)         |
| LLaMA-33B     | [BoolQ](https://github.com/google-research-datasets/boolean-questions)    | 83.2%     | [83.1](https://paperswithcode.com/sota/question-answering-on-boolq)   | LLaMA-65B    | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 85.7%     | [86.6](https://paperswithcode.com/sota/question-answering-on-boolq) |
| LLaMA2-7B     | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                      | 45.7%     | 45.3%                                                                 | LLaMA2-13B   | [BoolQ](https://paperswithcode.com/dataset/boolq)                      | 82.2%     | [81.7](https://paperswithcode.com/sota/question-answering-on-boolq) |
| LLaMA2-34B    | [BoolQ](https://github.com/google-research-datasets/boolean-questions)    | 85.9%     | --                                                                    | LLaMA2-70B   | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 65.1%     | --                                                                  |
| LLaMA3-8B     | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                      | 65.3%     | 66.6%                                                                 | LLaMA3-70B   | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 78.3%     | 79.5%                                                               |
| LLaMA3.1-8B   | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                      | 65.26%    | 66.7%                                                                 | LLaMA3.1-70B | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                | 81.8%     | 79.3%                                                               |
| Mistral-7B    | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                       | 56.3%     | 56.3%                                                                 | Mixtral-8x7B | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                    | 69.9%     | [70.6%](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu)                                                               |
| Mistral-8x22B | [MMLU](https://paperswithcode.com/dataset/mmlu)    | 77%       | [77.8%](https://mistral.ai/news/mixtral-8x22b/)                                                                 | --           | --                                                                     | --        | --                                                                  |
| QWen-7B       | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                      | 58.1%     | [58.2%](https://huggingface.co/Qwen/Qwen-7B)                          | Qwen-14B     | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                    | 65.3%     | [66.3%](https://huggingface.co/Qwen/Qwen-14B)                       |
| QWen-72B      | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                       | 74.6%     | [77.4%](https://huggingface.co/Qwen/Qwen-72B)                         | QWen1.5-0.5B | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                   | 31.8%     | 31.8%                                                               |
| QWen1.5-1.8b  | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                      | 46.2%     | [46.8%](https://qwenlm.github.io/zh/blog/qwen1.5/)                    | QWen1.5-4B   | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 55.0%     | [0.561](https://qwenlm.github.io/zh/blog/qwen1.5)                   |
| QWen1.5-7B    | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                       | 60.3%     | [61.0%](https://qwenlm.github.io/zh/blog/qwen1.5/)                    | QWen1.5-14B  | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                    | 67.3%     | [67.6%](https://qwenlm.github.io/zh/blog/qwen1.5)                   |
| QWen1.5-32B   | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                       | 72.6%     | [73.4%](https://huggingface.co/Qwen/Qwen-72B)                         | QWen1.5-72B  | [MMLU](https://paperswithcode.com/dataset/mmlu)                                                                    | 77.5%     | [77.5%](https://qwenlm.github.io/zh/blog/qwen1.5)                   |
| Qwen1.5-110B  | [MMLU](https://paperswithcode.com/dataset/mmlu)                           | 80.4%     | [80.4%](https://qwenlm.github.io/zh/blog/qwen1.5-110b/)               | Yi-34B       | [MMLU](https://paperswithcode.com/dataset/mmlu)                         | 76.3%     | [75.8%](https://hub.opencompass.org.cn/dataset-detail/MMLU)         |
| Qwen2-0.5B    | [MMLU](https://paperswithcode.com/dataset/mmlu)                           | 44.6%     | [45.4%](https://qwenlm.github.io/zh/blog/qwen2/)                      | Qwen2-1.5B   | [MMLU](https://paperswithcode.com/dataset/mmlu)                         | 54.7%     | [56.5%](https://qwenlm.github.io/zh/blog/qwen2/)                    |
| QWen2-7B      | [MMLU](https://paperswithcode.com/dataset/mmlu)                           | 70.3%     | [70.3%](https://qwenlm.github.io/zh/blog/qwen2/)                      | Qwen2-72B    | [MMLU](https://paperswithcode.com/dataset/mmlu)                         | 83.6%     | [84.2%](https://qwenlm.github.io/zh/blog/qwen2/)                    |
| CodeQwen1.5-7B | [Human Eval](https://huggingface.co/datasets/openai_humaneval)            | 54.8%     | [51.8%](https://qwenlm.github.io/zh/blog/codeqwen1.5/)                      |
MiniCPM-2B    | [MMLU](https://paperswithcode.com/dataset/mmlu)                           | 51.6%     | [53.4%](https://github.com/OpenBMB/MiniCPM?tab=readme-ov-file#3)                      | MiniCPM-MoE-8x2B | [BoolQ](https://github.com/google-research-datasets/boolean-questions)                      | 83.9%     | --                                                                  |



---


## 社区BUG列表


1. Baichuan-13B: 在任务执行过程中如果出现报错：AttributeError: 'BaichuanTokenizer’ object has no attribute 'sp_model'，请执行下面命令解决这个问题：

    ```shell
    pip install transformers==4.32.0 --force
    ```

2. GPT: GPT词表文件与常规模型不同：

    ```shell
    mkdir vocab_file 
    cd vocab_file
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    cd ..

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

    # 请根据真实存放路径配置预训练脚本以下参数
    VOCAB_FILE="./vocab_file/gpt2-vocab.json"   # 词表
    MERGE_FILE="./vocab_file/gpt2-merges.txt"   # BPE 合并表
    DATA_PATH="./dataset/gpt_text_sentence"     # 数据路径
    ```

3. Bloom-176B: config.json中同字段对应的key值与其他模型不一致，将文件中的n_embed改为hidden_size， 将num_attention_heads修改为n_head


4. QWen: 不包含QWen1.5等，需要修改权重文件 

    ```shell
   # 修改modelling_qwen.py文件第39行，将：
   # SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
   # 修改为：
   # SUPPORT_FP16 = True
    ```
