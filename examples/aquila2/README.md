# Aquila2  $\color{black}{\bf\tiny{【社区贡献模型】}}$
<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
    </p>
</p>

- [Aquila2-7B](#aquila2-7b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
  - [推理](#推理)
  - [评估](#评估)

# Aquila2-7B
## 训练

Aquila2-7B 训练的硬件配置如下:


|  硬件 |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器并切换到modellink分支

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout -f bcce6f
    cp -r megatron ../ModelLink/
    cd ..
    cd ModelLink
    mkdir logs
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    ```

2. 搭建conda环境

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test
    
    # 通过 PTA 包提供的 whl 安装 torch、torch_npu 和 apex，例如：
    pip install torch-2.2.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    pip install torch_npu-2.2.0*-cp38-cp38-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38-linux_aarch64.whl

    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitee.com/ascend/AscendSpeed.git
    cd AscendSpeed/
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
    pip install -r requirements.txt
    pip install -e .
    cd ..

    # 安装其余依赖包
    pip install -r requirements.txt
    ```

3. 使用浏览器下载 [Aquila2-7B模型的配置，tokenizer，和预训练权重](https://huggingface.co/BAAI/Aquila2-7B/tree/main)

    保存在 ModelLink/model_from_hf/Aquila2-7B/ 目录。

4. 权重转换

    将模型权重文件从 HuggingFace权重 格式转化为 Megatron 权重
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --load-dir ./model_from_hf/Aquila2-7B/ \
        --save-dir ./model_weights/Aquila2-7B-v0.1-tp8-pp1/ \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --tokenizer-model ./model_from_hf/Aquila2-7B/tokenizer.json
    ```

    任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
    ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Aquila2-7B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/Aquila2-7B/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Aquila2-7B/mg2hg/
    ```

    权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。

5. 预训练

    5.1 准备数据集

    下载 Aquila2-7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # 处理数据   
    mkdir -p ./dataset/Aquila2-7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Aquila2-7B/ \
        --output-prefix ./dataset/Aquila2-7B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 预训练

    配置 Aquila2-7B 训练脚本: examples/aquila2/pretrain_aquila2_7b_ptd.sh

    ```shell
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    TOKENIZER_PATH="./model_from_hf/Aquila2-7B/"  #tokenizer 路径
    DATA_PATH="./dataset/Aquila2-7B/alpaca_text_document"  #数据集 路径
    CKPT_LOAD_DIR="./model_weights/Aquila2-7B-v0.1-tp8-pp1/"
    CKPT_SAVE_DIR="./ckpt/Aquila2-7B/"
    ```

    - 如果不需要加载权重，就不需要设置CKPT_LOAD_DIR, 并且启动脚本里应不使用 `--load` 参数
    - 如果不需要保存权重，就不需要设置CKPT_SAVE_DIR, 并且启动脚本里应不使用 `--save` 参数
    - 进行断点续训时，应先按以上save的场景配置，待完成ckpt保存后，再修改相应参数，按以上load的场景加载已保存的ckpt。

    启动 Aquila2-7B 预训练脚本: examples/aquila2/pretrain_aquila2_7b_ptd.sh

    ```shell
    bash examples/aquila2/pretrain_aquila2_7b_ptd.sh
    ```

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

6. 微调

    6.1 准备微调数据集
    下载微调数据集 [这里](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据集
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # 处理微调数据集  
    mkdir ./finetune_dataset/Aquila2-7B/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Aquila2-7B/ \
        --output-prefix ./finetune_dataset/Aquila2-7B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 全参微调
    全参微调的配置脚本基本和预训练脚本 `pretrain_aquila2_7b_ptd.sh` 一致. *区别是数据集，以及增加训练参数`--is-instruction-dataset`*

    增加微调参数`--finetune`，使微调从第一步开始。

    ```bash
    DATA_PATH="./finetune_dataset/Aquila2-7B/alpaca"
    CKPT_LOAD_DIR="./ckpt/Aquila2-7B/"
        --load ${CKPT_LOAD_DIR} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```

### 性能

#### 吞吐

Aquila2-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 | 硬件           | 模型       | 迭代数| token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) |
|------|---------------|------------|------|------------------------|----------------------|
| NPU  | 910b 1node*8p | Aquila2-7B  | 2000 | 3323                  | 4.93                  | 
| 参考  |              | Aquila2-7B  | 2000 | 2673                  |    6.13               | 

## 推理

我们支持使用 Aquila2-7B进行文本生成的推理。

推理与预训练不同，我们必须加载预训练权重，请注意：在转换权重时使用的模型结构参数，和运行评估任务时使用的模型结构参数，应保持一致。

权重转换完成后，我们配置Aquila2-7B推理脚本`examples/aquila2/generate_aquila2_7b_ptd.sh`，需要正确指定加载权重的路径，词表路径等（下面样例仅供参考）

```shell
# 请按实际情况修改模型权重路径和分词器路径
CKPT_LOAD_DIR="./model_weights/Aquila2-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Aquila2-7B/"
```

启动Aquila2-7B推理:

```shell
bash examples/aquila2/generate_aquila2_7b_ptd.sh
```

部分推理样例如下：

Aquila2-7B:

![Inference](../../sources/images/aquila2/aquila2-7b-generate.png)

## 评估

我们使用 BoolQ benchmark 来评估我们的模型。在[Benchmark下载页面](https://github.com/google-research-datasets/boolean-questions)找到[数据集](https://storage.cloud.google.com/boolq/dev.jsonl)下载后保存。例如，保存在ModelLink/boolq/test目录下。

评估与推理类似，也必须加载转换后的权重，请注意：在转换权重时使用的模型结构参数，和运行评估任务时使用的模型结构参数，应保持一致。

权重转换完成后，我们配置Aquila2-7B评估脚本 `examples/aquila2/evaluate_aquila2_7b_ptd.sh`，需要正确指定加载权重的路径，词表路径，评估数据的路径，以及评估任务的名字等(下面样例仅供参考)：

```shell
CKPT_LOAD_DIR="./model_weights/Aquila2-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Aquila2-7B/"
EVAL_DATA_PATH="./boolq/test"
TASK="boolq"
```

启动Aquila2-7B评估

```shell
bash examples/aquila2/evaluate_aquila2_7b_ptd.sh
```

Aquila2-7B在**Ascend NPU**中的评测表现：

| 任务                                                                   | 模型       | 昇腾值|社区值|
|------------------------------------------------------------------------|------------|--------|------|
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Aquila2-7B  | 77.3% | 77.6% |
