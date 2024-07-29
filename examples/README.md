# ModelLink 开发使用指南

<p align="left">
</p>

## 目录


- [环境安装](#环境安装)
- [大模型预训练](#大模型预训练)

## 环境安装
    
【模型开发时推荐使用配套的环境版本】

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          driver           |         在研版本          |
|         firmware          |         在研版本          |
|           CANN            |             在研版本             |
|           torch           |            2.1.0、2.2.0             |
|         torch_npu         |           在研版本           |


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
    git checkout 2b0edd2
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..
    
    # 安装其余依赖库
    pip install -r requirements.txt 
```

## 大模型预训练

#### 1. 准备工作
配置脚本前需要完成前置准备工作，包括：**环境安装**、**数据集准备及处理**、**Huggingface权重转换**，详情可查看对应章节

#### 2. 配置预训练参数

legacy分支的预训练脚本保存在 example 中各模型文件夹下：pretrain_xxx_xx.sh
 
mcore分支的预训练脚本保存在 example/mcore 中各模型文件夹下：pretrain_xxx_xx.sh

需根据实际情况修改路径和参数值：

**示例：** examples/mcore/grok1/pretrain_grok1_40b_ptd.sh *(以grok1-40B为例)*

路径配置：包括**权重保存路径**、**权重加载路径**、**词表路径**、**数据集路径**
 ```shell
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/grok1-40B/"  #模型参数保存路径
    CKPT_LOAD_DIR="./model_weights/grok1-40B/权重文件"  #权重加载路径
    TOKENIZER_MODEL="./model_from_hf/grok1-40B/tokenizer.model"  #词表路径
    DATA_PATH="./dataset/grok1-40B/alpaca_text_document"  #数据集路径
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
    --data-path 0.5 "./dataset/grok1-40B/alpaca_text_document1" 0.5 "./dataset/grok1-40B/alpaca_text_document2"
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
【mcore分支】 
```shell
    bash example/mcore/模型文件夹/pretrain_xxx_xxx.sh
```

**示例：** *(以grok1-40B为例)*
```shell
    bash examples/mcore/grok1/pretrain_grok1_40b_ptd.sh
```
**注意**：
- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据


    
