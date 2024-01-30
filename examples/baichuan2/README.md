# BaiChuan2
<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/ModelLink/blob/modellink/examples/baichuan2/README_en.md">English</a> </b> 
    </p>
</p>


#  目录
- [Baichuan2-7B](#Baichuan2-7B)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
        - [吞吐](#吞吐)
        - [精度](#精度)


# Baichuan2-7B

## 训练
Baichuan2-7B 训练的硬件配置如下：

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
pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_XXX.whl

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

从 [huggingface](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main) 下载预训练权重：

```shell
mkdir baichuan2-7B-hf
cd ./baichuan2-7B-hf
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/generation_utils.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model-00001-of-00002.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model-00002-of-00002.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/quantizer.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenizer.model
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenizer_config.json
cd ..
```

接着将hf格式的权重转化为AscendSpeed可以加载的形式：
```shell
mkdir baichuan2-7B-mt

SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
# for ptd
python $SCRIPT_PATH \
    --input-model-dir ./baichuan2-7B-hf \
    --output-model-dir ./baichuan2-7B-mt \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --type 7B \
    --merge-mlp \
    --pse  
```


4. 准备数据集

从 [这里](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 下载 Baichuan2-7B-Base 的数据集：

```shell
# 下载数据集
mkdir dataset_baichuan2-7B
cd ./dataset_baichuan2-7B
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

# 准备数据集                              
python ./tools/preprocess_data.py \
--input ./dataset_baichuan2-7B/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
--tokenizer-name-or-path ./baichuan2-7B-hf \
--output-prefix ./dataset_baichuan2-7B/alpaca \
--workers 4 \
--log-interval 1000 \
--tokenizer-type PretrainedFromHF
```


5. 配置 Baichuan2-7B 预训练脚本: examples/baichuan/pretrain_baichuan2_ptd_7B.sh 

```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改数据集，权重，词表等路径
CKPT_SAVE_DIR="./ckpt"
DATA_PATH="./dataset_baichuan2-7B/alpaca_text_document"
TOKENIZER_MODEL="./baichuan2-7B-hf/tokenizer.model"
CKPT_LOAD_DIR="./baichuan2-7B-mt"
```

6. 启动 Baichuan2-7B 预训练脚本: examples/baichuan2/pretrain_baichuan2_ptd_7B.sh 

```shell
bash examples/baichuan2/pretrain_baichuan2_ptd_7B.sh 
```

### 性能

#### 吞吐

Baichuan2-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|  设备  |    模型     | 迭代数  | 样本吞吐 (samples/s) | tokens吞吐 (tokens/s/p) | 单步迭代时间 (s/step) | 
|:----:|:---------:|:----:|:---------------------:|:---------------:|:----------------:|
| NPUs | Baichuan2-7B | 1000 | 4.59 | 2349 | 6.973| 
|  参考  | Baichuan2-7B | 1000 | 5.40 |  2769 | 5.915       | 



#### 精度

NPU vs 参考 loss.

![NPU-LOSS](../../sources/images/baichuan2/baichuan2-7B-loss-compare.png)

NPU vs 参考 loss 相对误差.

![NPU-Relative-Error](../../sources/images/baichuan2/baichuan2-7B-loss-relative-error.png)


