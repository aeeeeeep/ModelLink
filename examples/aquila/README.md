# Aquila-7B
简体中文 | [English](https://gitee.com/ascend/ModelLink/blob/modellink/examples/aquila/README_en.md)

- [Aquila-7B](#Aquila-7B)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
  - [推理](#推理)
  - [评估](#评估)

## 训练

Aquila-7B 训练的硬件配置如下:


|  硬件 |  设备 |        配置       |
|:---:|:---:|:---------------:|
| NPU | A+K | 8 x Ascend NPUs |

### 脚本

1. 克隆git仓库到个人服务器并切换到modellink分支
```shell
git clone https://gitee.com/ascend/ModelLink.git
cd ModelLink
mkdir logs
mkdir ckpt
```

2. 搭建conda环境

```bash
# python3.8
conda create -n test python=3.8
conda activate test
# 通过互联网上提供的pip源安装 torch，可能需要尝试合适的包含这个torch==2.1.0版本的pip源
pip install torch==2.1.0
# 通过PTA上提供的安装包，以whl文件方式安装aarch64架构上的2.1.0版本的torch_npu
pip install torch_npu-2.1.0.postxxxx-cp38-cp38-xxxx_aarch64.whl
# 通过PTA上提供的安装包，以whl文件方式安装apex
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# 拉取AscendSpeed源代码，进入AscendSpeed目录，source相应的CANN环境文件，然后源码方式安装ascendspeed加速包
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed/
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -e ./

# 进入ModelLink主目录，安装其余依赖包
cd ModelLink/
pip install -r requirements.txt
```


3. 使用浏览器下载 [Aquila-7B模型的配置，tokenizer，和预训练权重](https://huggingface.co/BAAI/Aquila-7B/tree/main)

保存在 ModelLink/HF_Aquila7B_downloaded/ 目录。


4. 数据预处理

第一步，使用浏览器 [下载数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)， 保存在ModelLink/dataset/ 目录

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

第二步，使用Aquila-7B指定的tokenizer处理数据集：

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ModelLink/
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./HF_Aquila7B_downloaded \
    --output-prefix ./dataset/aquila \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF
```

5. 权重转换

将模型权重文件从 huggingface 格式转化为 AscendSpeed 格式

```shell
cd ModelLink/
mkdir model_weights
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py \
    --model-type GPT \
    --load-dir ./HF_Aquila7B_downloaded \
    --save-dir ./model_weights/aquila \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --tokenizer-model ./HF_Aquila7B_downloaded/tokenizer.json
```


6. 配置 Aquila-7B 预训练脚本

需要在预训练脚本中配置相关参数
```shell
# 修改数据集路径，权重路径，词表路径等
TOKENIZER_PATH=./HF_Aquila7B_downloaded  #tokenizer 路径
DATA_PATH=./dataset/aquila_text_document  #数据集 路径
CKPT_LOAD_DIR=./model_weights/aquila
CKPT_SAVE_DIR=./ckpt
# 如果不需要加载权重，就不需要设置CKPT_SAVE_DIR, 并且启动脚本里应不使用 `--load` 参数
```

7. 启动 Aquila-7B 预训练脚本

运行预训练脚本前，需先执行set_env.sh脚本以便设置环境参数，或者也可将其放入预训练脚本中执行。
```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

按以下方式启动Aquila-7B预训练：
```shell
bash examples/aquila/pretrain_aquila_7b_ptd.sh
```

### 性能

#### 吞吐
Aquila-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 | 硬件           | 模型       | 迭代数| token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) |
|------|---------------|------------|------|------------------------|----------------------|
| NPU  | 910b 1node*8p | Aquila-7B  | 1000 | 2849                  | 5.75                  | 
| 参考  |              | Aquila-7B  | 1000 | 2874                   |    5.70               | 



## 推理

我们支持使用 Aquila-7B进行文本生成的推理。

推理与预训练不同，我们必须加载预训练权重，请注意：在转换权重时使用的模型结构参数，和运行评估任务时使用的模型结构参数，应保持一致。

权重转换完成后，我们配置Aquila-7B推理脚本`tasks/inference/generate_aquila_7b_ptd.sh`，需要正确指定加载权重的路径，词表路径等（下面样例仅供参考）

```shell
# 请按实际情况修改模型权重路径和分词器路径
CKPT_LOAD_DIR="./model_weights/aquila/"
TOKENIZER_PATH="./HF_Aquila7B_downloaded/"
```

启动Aquila-7B推理:
```shell
bash ./tasks/inference/generate_aquila_7b_ptd.sh
```

部分推理样本如下：

Aquila-7B:

![aquila-7B_generate.png](../../sources/images/aquila/aquila_7B_generate_ptd_0205.png)


## 评估

我们使用 BoolQ benchmark 来评估我们的模型。在[Benchmark下载页面](https://github.com/google-research-datasets/boolean-questions)找到[数据集](https://storage.cloud.google.com/boolq/dev.jsonl)下载后保存。例如，保存在ModelLink/boolq/test目录下。

评估与推理类似，也必须加载转换后的权重，请注意：在转换权重时使用的模型结构参数，和运行评估任务时使用的模型结构参数，应保持一致。

权重转换完成后，我们配置Aquila-7B评估脚本 `tasks/evaluation/evaluate_aquila_7b_ptd.sh`，需要正确指定加载权重的路径，词表路径，评估数据的路径，以及评估任务的名字等(下面样例仅供参考)：

```shell
CKPT_LOAD_DIR="./model_weights/aquila/"
TOKENIZER_PATH="./HF_Aquila7B_downloaded/"
EVAL_DATA_PATH="./boolq/test"
TASK="boolq"
```

启动Aquila-7B评估
```shell
bash tasks/evaluation/evaluate_aquila_7b_ptd.sh
```

Aquila-7B在**Ascend NPU**中的评测表现：

| 任务                                                                   | 模型       | 昇腾值|社区值|
|------------------------------------------------------------------------|------------|--------|------|
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Aquila-7B  | 77.3% |      |
