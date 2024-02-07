# BaiChuan
<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/ModelLink/blob/modellink/examples/baichuan/README_en.md">English</a> </b> 
    </p>
</p>



#  目录

- [Baichuan-7B](#Baichuan-7B)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
        - [吞吐](#吞吐)
        - [精度](#精度)
- [Baichuan-13B](#Baichuan-13B)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
        - [吞吐](#吞吐)
        - [精度](#精度)
  - [推理](#推理)
  - [评估](#评估)

# Baichuan-7B

## 训练
Baichuan-7B 训练的硬件配置如下：

|  硬件 |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |

### 脚本

1. 拷贝仓库到你的个人服务器：
```shell
git clone https://gitee.com/ascend/ModelLink.git 
cd ModeLlink 
git checkout modellink
mkdir logs
mkdir ckpt
```

2. 搭建环境

```bash
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu 
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装加速库
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt 
```

3. （可选）准备预训练权重

从 [huggingface](https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main) 下载预训练权重：



```shell
mkdir baichuan-7B-hf
cd ./baichuan-7B-hf
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/generation_config.json
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/handler.py
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/pytorch_model.bin
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenizer.model
wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenizer_config.json
cd ..
```

接着将hf格式的权重转化为AscendSpeed可以加载的形式：
```shell
mkdir baichuan-7B-mt

SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
    --input-model-dir ./baichuan-7B-hf \
    --output-model-dir ./baichuan-7B-mt \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --type 7B \
    --pse \
    --merge-mlp
```


4. 准备数据集

从 [这里](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 下载 BaiChuan-7B 的数据集：

```shell
# 下载数据集
mkdir dataset_baichuan7B
cd ./dataset_baichuan7B
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

# 准备数据集                              
python ./tools/preprocess_data.py \
--input ./dataset_baichuan7B/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
--tokenizer-name-or-path ./baichuan-7B-hf \
--output-prefix ./dataset_baichuan7B/alpaca \
--workers 4 \
--log-interval 1000 \
--tokenizer-type PretrainedFromHF
```


5. 配置 Baichuan-7B 预训练脚本: examples/baichuan/pretrain_baichuan_ptd_7B.sh 

```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

CKPT_SAVE_DIR="./ckpt"
DATA_PATH="./dataset_baichuan7B/alpaca_text_document"
TOKENIZER_MODEL="./baichuan-7B-hf/tokenizer.model"
CKPT_LOAD_DIR="./baichuan-7B-mt"
```

6. 启动 Baichuan-7B 预训练脚本: examples/baichuan/pretrain_baichuan_ptd_7B.sh 

```shell
bash examples/baichuan/pretrain_baichuan_ptd_7B.sh 
```

### 性能

#### 吞吐

Baichuan-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|  设备  |    模型     | 迭代数  | 样本吞吐 (samples/s) | tokens吞吐 (tokens/s/p) | 单步迭代时间 (s/step) | 
|:----:|:---------:|:----:|:---------------------:|:---------------:|:----------------:|
| NPUs | Baichuan-7B | 1000 | 5.16 | 2643.00 | 6.199| 
|  参考  | Baichuan-7B | - | - |  2036 | - | 

#### 精度

NPU vs 参考 loss.

![NPU-LOSS](../../sources/images/baichuan/baichuan7B-loss-compare.png)

NPU vs 参考 loss 相对误差.

![NPU-Relative-Error](../../sources/images/baichuan/baichuan7B-loss-relative-error.png)


# Baichuan-13B

## 训练

Baichuan-13B 训练的硬件配置如下:

|  硬件 |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |



### 脚本

1. 拷贝仓库到你的个人服务器

```shell
git clone https://gitee.com/ascend/ModelLink.git 
cd ModeLlink 
git checkout modellink
mkdir logs
mkdir ckpt
```

2. 搭建环境

```bash
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu 
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装加速库
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt 
```

3. （可选的）准备预训练权重

从 [huggingface](https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main) 下载预训练权重
```shell
mkdir baichuan-13B-hf
cd ./baichuan-13B-hf
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/generation_config.json
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/pytorch_model-00001-of-00003.bin
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/pytorch_model-00002-of-00003.bin
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/pytorch_model-00003-of-00003.bin
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/quantizer.py
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/tokenizer_config.json
wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/tokenizer.model
cd ..
```

将 BaiChuan-13B 模型权重从 huggingface 格式转换为 AscendSpeed 格式
```shell
mkdir baichuan-13B-mt

SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
    --input-model-dir ./baichuan-13B-hf \
    --output-model-dir ./baichuan-13B-mt \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --type 13B \
    --pse  \
    --merge-mlp   
```

4. 准备数据集

下载 Baichuan-13B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
mkdir dataset_baichuan13B
cd ./dataset_baichuan13B
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

python ./tools/preprocess_data.py \
    --input ./dataset_baichuan13B/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./baichuan-13B-hf \
    --output-prefix ./dataset_baichuan13B/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF 
```


5. 配置 Baichuan-13B 训练脚本: examples/baichuan/pretrain_baichuan_ptd_13B.sh


```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

CKPT_SAVE_DIR="./ckpt"
DATA_PATH="./dataset_baichuan13B/alpaca_text_document"
TOKENIZER_MODEL="./baichuan-13B-hf/tokenizer.model"
CKPT_LOAD_DIR="./baichuan-13B-mt" 
```

6. 启动 Baichuan-13B 训练脚本: examples/baichuan/pretrain_baichuan_ptd_13B.sh

```bash
bash examples/baichuan/pretrain_baichuan_ptd_13B.sh
```


### 性能

#### 吞吐

Baichuan-13B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比:

|  设备  |      模型      | 迭代数  | 样本吞吐 (samples/s) | token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 
|:----:|:------------:|:----:|:------------------:|:--------------------:|:---------------:|
| NPUs | Baichuan-13B | 1000 | 2.37 | 1213| 13.5 |      
|  参考  | Baichuan-13B | - |  -   |         862          |     -   |    


#### 精度

NPU vs 参考 loss.


![NPU-LOSS](../../sources/images/baichuan/baichuan13B-loss-compare.png)

NPU vs 参考 loss 相对误差.

![NPU-Relative-Error](../../sources/images/baichuan/baichuan13B-loss-relative-error.png)


## 推理

首先需要配置baichuan-13B的推理脚本: tasks/inference/generate_baichuan_13b_ptd.sh

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 修改模型权重路径和词表路径
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
```

然后可直接启动generate_baichuan_13b_ptd.sh

```bash
bash tasks/inference/generate_baichuan_13b_ptd.sh
```

推理的示例如下:
![Inference](../../sources/images/baichuan/baichuan_13B_inference.png)

## 评估

我们使用boolq基准来评估我们的模型。基准[下载](https://super.gluebenchmark.com/tasks).
1. 准备数据集
```shell
 mkdir boolq
 cd boolq
 wget https://storage.googleapis.com/boolq/dev.jsonl
 cd ..
```
2. 配置Baichuan-13B评估脚本: tasks/evaluation/eval_baichuan_13B.sh

```shell
# 配置原始权重与词表的路径
CHECKPOINT=<origin-ckpt-path>
TOKENIZER_PATH=<tokenizer-path>
# 配置任务以及数据路径
DATA_PATH="./boolq/"
TASK="boolq"
```
3. 执行Baichuan-13B评估脚本: tasks/evaluation/eval_baichuan_13B.sh

```shell
bash ./tasks/evaluation/eval_baichuan_13B.sh
```

<table>
  <thead>
    <tr>
      <th>任务</th>
      <th>验证集</th>
      <th>模型</th>
      <th>昇腾值</th>
      <th>社区值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">Boolq</a></td>
      <td>test</td>
      <th>Baichuan 13B</th>
      <td>0.747</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/BoolQ">0.736</a></td>
    </tr>
  </tbody>
</table>
