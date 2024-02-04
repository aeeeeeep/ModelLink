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
pip install torch-2.1.0-cp37-cp37m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp37-cp37m-linux_aarch64.whl
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
| NPUs | Baichuan-7B | 1000 | 4.78 | 2448.76 | 6.688| 
|  参考  | Baichuan-7B | 1000 | 5.45 |  2792.56 | 5.863       | 

#### 精度

NPU vs 参考 loss.

![NPU-LOSS](../../sources/images/baichuan/baichuan7B-loss-compare.png)

NPU vs 参考 loss 相对误差.

![NPU-Relative-Error](../../sources/images/baichuan/baichuan7B-loss-relative-error.png)



