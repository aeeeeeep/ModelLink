# README

## 变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| cur_dir | 运行指令或执行脚本时的路径（当前目录）                  |
| version | 版本                  |

## 环境准备
### 依赖版本
- 模型仓代码配套可运行的硬件型号
  - Atlas 800I A2（32GB显存）
  - Atlas 300I DUO（96GB显存）
- 模型仓代码运行相关配套软件
  - 系统OS
  - 驱动（HDK）
  - CANN
  - Python
  - PTA
- 版本配套关系
  - 当前模型仓需基于CANN包8.0版本及以上，Python 3.10，torch 2.0.1进行环境部署与运行
  - 待正式商发后补充版本链接

### 1.1 安装HDK

详细信息可参见[昇腾社区驱动与固件](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/envdeployment/instg/instg_000018.html)，先安装firmwire，再安装driver

#### 1.1.1 安装firmwire

安装方法:

| 包名                                   |
|--------------------------------------|
| Ascend-hdk-*-npu-firmware_${version}.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装firmwire
chmod +x Ascend-hdk-*-npu-firmware_${version}.run
./Ascend-hdk-*-npu-firmware_${version}.run --full
```

#### 1.1.2 安装driver

安装方法：

| cpu     | 包名                                               |
|---------|--------------------------------------------------|
| aarch64 | Ascend-hdk-*-npu-driver_${version}_linux-aarch64.run |
| x86     | Ascend-hdk-*-npu-driver_${version}_linux-x86-64.run  |

```bash
# 根据CPU架构 以及npu型号 安装对应的 driver
chmod +x Ascend-hdk-*-npu-driver_${version}_*.run
./Ascend-hdk-*-npu-driver_${version}_*.run --full
```

### 1.2 安装CANN

详细信息可参见[昇腾社区CANN软件](https://www.hiascend.com/software/cann)，先安装toolkit 再安装kernel

#### 1.2.1 安装toolkit

安装方法：`${version}`代表具体版本

| cpu     | 包名                                            |
|---------|-----------------------------------------------|
| aarch64 | Ascend-cann-toolkit_${version}_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_${version}_linux-x86_64.run  |

```bash
# 安装toolkit  以arm为例
chmod +x Ascend-cann-toolkit_${version}_linux-aarch64.run
./Ascend-cann-toolkit_${version}_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 1.2.2 安装kernel

安装方法：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-*_${version}_linux.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装 kernel
chmod +x Ascend-cann-kernels-*_${version}_linux.run
./Ascend-cann-kernels-*_${version}_linux.run --install
```

### 1.3 安装PytorchAdapter

先安装torch 再安装torch_npu

#### 1.3.1 安装torch

安装方法：

| 包名                                           |
|----------------------------------------------|
| torch-2.0.1+cpu-cp310-cp310-linux_x86_64.whl   |
| torch-2.0.1-cp310-cp10-linux_aarch64.whl      |
| ...                                          |

根据所使用的环境中的python版本以及cpu类型，选择对应版本的torch安装包。

```bash
# 安装torch 2.0.1 的python 3.10 的arm版本为例
pip install torch-2.0.1-cp310-cp310-linux_aarch64.whl
```

#### 1.3.2 安装torch_npu

[下载PyTorch Adapter](https://www.hiascend.com/software/ai-frameworks/commercial)，安装方法：

| 包名                          |
|-----------------------------|
| pytorch_v2.0.1_py310.tar.gz  |
| pytorch_v2.0.1_py310.tar.gz  |
| ...                         |

- 安装选择与torch版本 以及 python版本 一致的npu_torch版本

```bash
# 安装 torch_npu 以torch 2.0.1 的python 3.10的版本为例
tar -zxvf pytorch_v2.0.1_py310.tar.gz
pip install torch*_aarch64.whl
```

### 1.4 安装加速库
- 下载加速库
  - 加速库下载链接待补充
 
  | 包名                          |
  |-----------------------------|
  | Ascend-mindie-atb_1.0.RC1_linux-aarch64_abi0.run  |
  | Ascend-mindie-atb_1.0.RC1_linux-aarch64_abi1.run  |
  | Ascend-mindie-atb_1.0.RC1_linux-x86_64_abi0.run |
  | Ascend-mindie-atb_1.0.RC1_linux-x86_64_abi1.run |
  | ...                         |

  - 将文件放置在\${working_dir}路径下

- 安装
    ```shell
    chmod +x Ascend-mindie-atb_*_linux-*_abi*.run
    ./Ascend-mindie-atb_*_linux-*_abi*.run --install --install-path=${working_dir}
    source ${working_dir}/atb/set_env.sh
    ```
- 可以使用`uname -a`指令查看服务器是x86还是aarch架构
- 可以使用以下指令查看abi是0还是1
    ```shell
    python -c "import torch; print(torch.compiled_with_cxx11_abi())"
    ```
    - 若输出结果为True表示abi1，False表示abi0

### 1.5 安装模型仓
- 场景一：使用编译好的包进行安装
  - 下载编译好的包
    - 下载链接待补充
 
    | 包名                          |
    |-----------------------------|
    | Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torch1.11.0-abi0.tar.gz  |
    | Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torch2.0.1-abi1.tar.gz  |
    | Ascend-mindie-atb-models_1.0.RC1_linux-x86_64_torch1.11.0-abi1.tar.gz |
    | Ascend-mindie-atb-models_1.0.RC1_linux-x86_64_torch2.0.1-abi1.tar.gz |
    | ...                         |

  - 将文件放置在\${working_dir}路径下
  - 解压
    ```shell
    cd ${working_dir}
    mkdir ModelLink
    cd ModelLink
    tar -zxvf ../Ascend-mindie-atb-models_*_linux-*_torch*-abi*.tar.gz
    ```
  - 安装atb_llm whl包
    ```
    cd ${working_dir}/ModelLink
    pip install atb_llm-0.0.1-py3-none-any.whl
    ```
- 场景二：手动编译模型仓
  - 获取模型仓代码
    ```shell
    git clone https://gitee.com/ascend/ModelLink.git
    ```
  - 切换到目标分支
    ```shell
    cd ModelLink
    git checkout master
    ```
  - 代码编译
    ```shell
    cd mindie_ref/mindie_llm/atb_models
    bash scripts/build.sh
    cd output/atb/
    source set_env.sh
    ```

- 安装 sdk

  ```
  cd ${working_dir}/ModelLink/pytorch/examples/atb_speed_sdk
  pip3 install .
  ```

## 环境变量参考

### CANN、加速库、模型仓的环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ${working_dir}/atb/set_env.sh
# 若使用编译好的包（即1.5章节的场景一），则执行以下两个指令
source ${working_dir}/ModelLink/set_env.sh
export PYTHONPATH=${working_dir}/ModelLink/:$PYTHONPATH
# 若使用gitee上的源码进行编译（即1.5章节的场景二），则执行以下两个指令
source ${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models/scripts/set_env.sh
export PYTHONPATH=${working_dir}/ModelLink/mindie_ref/mindie_llm/atb_models/:$PYTHONPATH
```

### 日志打印（可选）
- 加速库日志
  - 打开加速库日志
    ```shell
    export ATB_LOG_TO_FILE=1
    export ATB_LOG_TO_STDOUT=1
    export ATB_LOG_LEVEL=INFO
    export TASK_QUEUE_ENABLE=1
    export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
    ```
  - 关闭加速库日志
    ```shell
    export ATB_LOG_TO_FILE=0
    export ATB_LOG_TO_STDOUT=0
    export TASK_QUEUE_ENABLE=0
    export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
    ```
  - 日志存放在${cur_dir}/atb_temp/log下

- 算子库日志
  - 打开算子库日志
    ```shell
    export ASDOPS_LOG_TO_FILE=1
    export ASDOPS_LOG_TO_STDOUT=1
    export ASDOPS_LOG_LEVEL=INFO
    ```
  - 关闭算子库日志
    ```shell
    export ASDOPS_LOG_TO_FILE=0
    export ASDOPS_LOG_TO_STDOUT=0
    ```
  - 日志存放在~/atb/log下
- 注意：开启日志后会影响推理性能，建议默认关闭；当推理执行报错时，开启日志定位原因

### 性能Profiling
- 待补充

### Dump Tensor
- 适用于定位精度问题
- 使用方式见[dump tensor工具README](https://gitee.com/ascend/ait/tree/5ffd45a7f7520266976599912f8e35b97fb0c74d/ait/docs/llm#https://gitee.com/link?target=https%3A%2F%2Fais-bench.obs.cn-north-4.myhuaweicloud.com%2Fcompare%2F20231226%2FABI0%2Fait_llm-0.1.0-py3-none-linux_x86_64.whl)

## 特性支持矩阵
- 待补充

## 预置模型列表
- [BaiChuan](./examples/models/baichuan/README.md)
- [CodeGeeX2](./examples/models/codegeex/v2_6b/README.md)
- [ChatGLM2](./examples/models/chatglm/v2_6b/README.md)
- [LLaMa](./examples/models/llama/README.md)
- [Qwen](./examples/models/qwen/README.md)
- [StarCoder](./examples/models/starcoder/README.md)
- 多模态模型Readme链接待补充

## 问题定位
- 若遇到推理执行报错，优先打开日志环境变量，并查看算子库和加速库的日志中是否有error级别的告警，基于error信息进一步定位
- 若遇到精度问题，可以dump tensor后进行定位

## Key Feature
- 待补充

## Release Note
- 待正式商发后补充