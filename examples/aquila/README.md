# Aquila-7B
简体中文 | [English](https://gitee.com/ascend/ModelLink/blob/master/examples/aquila/README_en.md)

- [Aquila-7B](#Aquila-7B)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [精度](#精度)

## 训练

Aquila-7B 训练的硬件配置如下:


|  硬件 |  设备 |        配置       |
|:---:|:---:|:---------------:|
| NPU | A+K | 8 x Ascend NPUs |

### 脚本

#### 1. 克隆git仓库到个人服务器
```shell
git clone https://gitee.com/ascend/ModelLink.git
cd ModelLink
mkdir logs
mkdir ckpt
```

#### 2. 搭建conda环境

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
# 安装 deepspeed
pip install deepspeed==0.9.2
# 源码方式安装 deepspeed_npu
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install -e ./
cd ..
# 进入AscendSpeed目录，源码方式安装ascendspeed包
cd AscendSpeed/
pip install -e ./
# 进入ModelLink主目录，安装其余依赖包
cd ModelLink/
pip install -r requirements.txt
```


#### 3. 使用浏览器下载 [Aquila-7B模型的配置，tokenizer，和预训练权重](https://huggingface.co/BAAI/Aquila-7B/tree/main)

保存在 ModelLink/HF_Aquila7B_downloaded/ 目录。


#### 4. 数据预处理

第一步，使用浏览器 [下载数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)， 保存在ModelLink/dataset/ 目录

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

第二步，使用Aquila-7B指定的tokenizer处理数据集：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./HF_Aquila7B_downloaded/ \
    --output-prefix ./dataset/aquila \
    --workers 4 \
    --log-interval 1000  \
    --tokenizer-type PretrainedFromHF
```

#### 5. 权重转换

将模型权重文件从 huggingface 格式转化为 AscendSpeed 格式

```shell
mkdir model_weights
SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
    --input-model-dir ./HF_Aquila7B_downloaded/ \
    --output-model-dir ./model_weights/aquila \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --make-vocab-size-divisible-by 8 \
    --type 7B \
    --merge-mlp
```


#### 6. 配置 Aquila-7B 预训练脚本

```shell
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 修改数据集路径，权重路径，词表路径等
TOKENIZER_PATH=./HF_Aquila7B_downloaded  #tokenizer 路径
DATA=./dataset/aquila_text_document  #数据集 路径
CHECKPOINT=./model_weights/aquila

# 如果不需要加载权重，就移除 `--load` 参数
```

#### 7. 启动 Aquila-7B 预训练脚本

按以下方式启动训练：
Aquila-7B
```shell
bash examples/aquila/pretrain_aquila_7b_ptd.sh
```

### 性能
#### 精度

Aquila-7b NPU vs 参考 loss.

![NPU-GPU-Relative-Error](../../sources/images/aquila/aquila_comp0122.png)