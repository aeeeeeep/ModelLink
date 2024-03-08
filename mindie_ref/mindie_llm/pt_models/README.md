# 介绍

pt_models旨在在昇腾CANN框架下执行pytorch大模型脚本提供样例参考，方便开发者对自己的大模型进行NPU的迁移。

是一个快速且易于让大模型接入昇腾CANN框架的推理和服务库

性能优势体现：

- 

易用性优势体现：

- 

支持的Hugging Face大模型:

- llama2

# 公告

- 2024年3月6号：提供llama2-70b前端切分分布式执行样例(单batch)

# 环境依赖

|    软件    |             [版本](https://www.hiascend.com/zh/)             |
| :--------: | :----------------------------------------------------------: |
|   Python   |                            3.8.0                             |
|   driver   | [2023Q4商发](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/258924109?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
|  firmware  | [2023Q4商发](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/258924109?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
|    CANN    | [2023Q4商发](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/258923273?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
|   kernel   | [2023Q4商发](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/258923273?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
|   torch    |                            2.1.0                             |
| torch_npu  |   [2023Q4商发](https://gitee.com/ascend/pytorch/releases)    |
|    apex    | [2023Q4商发](https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.1.0/20231225.2/pytorch_v2.1.0_py38.tar.gz) |
| 第三方依赖 |                       requirement.txt                        |

# 环境搭建

```shell
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu
pip3 install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
pip3 install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

pip3 install -r requirements.txt 
```

[昇腾环境准备](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000005.html)

**注意**：建议昇腾run包安装在conda创建的镜像中安装

# 项目结构

```
│  llm_inference.py
│  llm_inference.sh  #启动脚本
│  README.md
│  requirement.txt
│  setup.sh
│  tree.txt
│      
├─common
│      utils.py
│      
├─config
│      llm_inference.json
│      
├─models
│  └─llama2  #基于源码对原始model结构的调整，主要是为了性能提升
│          modeling_llama.py
│          utils.py
│          
└─script
    └─llama2
            run_llama2.py
            __init__.py
```



# 模型及数据集





# 快速体验

```shell
cann_path=/usr/local/Ascend #昇腾cann包安装目录
model_path=xxx/llama2-70b #下载的对应模型的weight和tokenizer
bash setup.sh
bash llm_inference.sh --cann_path=${cann_path} --model_path=${model_path} --device_list="0,1,2,3,4,5,6,7"
```

# 执行参数介绍

| 参数             | 参数类型 | 参数说明                                                     |
| ---------------- | -------- | ------------------------------------------------------------ |
| model_path       | String   | 模型weight和tokenizer路径。必选                              |
| model            | String   | 期望执行的模型名，默认llama2。当前支持llama2                 |
| batch_size       | Int      | batch的大小，默认1                                           |
| seq_len_in       | Int      | 输入句子的最大长度，默认1024                                 |
| seq_len_out      | Int      | 输出句子的最大长度，默认1024                                 |
| dtype            | String   | 权重数据类型，默认fp16，支持fp16,fp32,bf16                   |
| device_list      | String   | 指定执行的device列表，默认0                                  |
| input_padding    | Bool     | 是否对输入padding到最大长度，默认false                       |
| exe_mode         | String   | pytorch脚本执行方式，默认图模式。支持eager和dynamo           |
| jit_compile      | Bool     | 是否进行算子编译，默认false，表示使能二进制，不做算子编译    |
| cann_path        | String   | CANN包安装路径，默认/usr/local/Ascend                        |
| distributed_mode | String   | 部署方式，默认deepspeed，分布式多卡执行。                    |
| log_level        | Int      | CANN日志级别，默认为3，支持0，1，2，3表示DEBUG,INFO,WARNING,ERROR |

# 加入用户自定义大模型

- 新增模型执行脚本

  在examples目录下创建自己大模型目录，并新增run_(model名).py的模型调用脚本，示例如下

  ```python
  
  ```

