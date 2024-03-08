# GLM-130B 推理指导

## 环境

### 设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/atb/set_env.sh
git clone https://gitee.com/ascend/Model_Link
bash ./Model_Link/speed_infer/scripts/build.sh
source ./Model_Link/speed_infer/output/atb_speed/set_env.sh
```

### 安装 Python 依赖
```shell
pip install -r requirements.txt
```
然后自行下载PTA包，安装`torch`与`torch_npu`。


## 推理准备

### 权重与测试数据

创建一个目录用于存放权重与测试数据：
```shell
mkdir -p /data/acltransformer_testdata
export ATB_TESTDATA=/data/acltransformer_testdata
```

按照 [THUDM/GLM-130B](https://github.com/THUDM/GLM-130B?tab=readme-ov-file#model-weights) 的说明下载并整理 [权重文件](https://docs.google.com/forms/d/e/1FAIpQLSehr5Dh_i3TwACmFFi8QEgIVNYGmSPwV0GueIcsUev0NEfUug/viewform?usp=sf_link)，然后下载 [测试数据集](https://cloud.tsinghua.edu.cn/f/826f0df4356f4022a264/) 并解压，最终将权重与数据集按以下结构存放：

```
├── ${ATB_TESTDATA}/
    ├── evaluation/
    │   ├── bloom/
    │   ├── CLUE/
    │   ├── lambada/
    │   └──  MMLU/
    └── weights/
        └── glm-130b-sat/
            ├── 49300/
            └── latest
```
使用量化工具生成量化权重（待补充）
```
export QUANT_PATH=/quant_path
```
### 模型代码
```shell
git clone https://github.com/THUDM/GLM-130B.git && cd GLM-130B
git checkout 212215c54f8a9da2deed51455868305235664370
git clone https://github.com/THUDM/SwissArmyTransformer.git sat && cd sat
git checkout 605523d8d9fb9acb09bbbab623acdd276695aa1d && cd ..
mv sat/SwissArmyTransformer . && rm -rf sat

cp -rf ../patchs/GLM-130B/* ./
cp -rf ../patchs/SwissArmyTransformer/* SwissArmyTransformer/
cp ../main.sh ../main.py ../input.txt ./
```

### 绑核
安装 numactl 工具，然后通过 `lspci -vs ${bus-id}` 查询每个 device 的 NUMA Node, 最后根据查询结果修改 main.sh 文件中的设备与 NUMA 节点的映射关系，以发挥最佳性能。


## 模型推理

### 精度评估
```shell
bash main.sh --evaluate --mode inference \
    --data-path ${ATB_TESTDATA}/evaluation \
    --task tasks/bloom/glue_cola.yaml tasks/mmlu/mmlu.yaml
```

测试结果如下：
|      | MMLU (Accuracy) | GLUE_COLA (Accuracy) |
| ---- | --------------- | -------------------- |
| FP16 | 44.702          | 57.411               |

### 性能评估
```shell
bash main.sh --benchmark --mode inference --atb_backend lccl
```
测试结果如下：
| batchsize | InputSeqLen(Encoding) | OutputSeqLen(Decoding) | FirstTokenTime(ms) | TimePerTokens(ms) |
| - | ---- | --- | ------- | ----- |
| 1 | 256  | 64  | 92.18   | 37.26 |
| 1 | 512  | 128 | 151.14  | 37.49 |
| 1 | 1024 | 256 | 274.20  | 37.98 |
| 1 | 1536 | 512 | 405.71  | 38.50 |
| 4 | 256  | 64  | 271.22  | 39.96 |
| 4 | 512  | 128 | 521.48  | 40.53 |
| 4 | 1024 | 256 | 1086.13 | 41.74 |
| 4 | 1536 | 512 | 1656.06 | 42.92 |
| 8 | 256  | 64  | 519.30  | 41.95 |
| 8 | 512  | 128 | 1071.61 | 42.90 |
| 8 | 1024 | 256 | 2224.83 | 44.71 |
| 8 | 1536 | 512 | 8897.74 | 46.78 |
