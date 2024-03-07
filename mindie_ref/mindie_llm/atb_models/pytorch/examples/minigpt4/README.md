# MiniGPT-4

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
    - [NPU 环境准备（HDK、CANN、PTA）](#NPU 环境准备（HDK、CANN、PTA）)
    - [推理环境准备（加速库、模型库）](#推理环境准备（加速库、模型库）)
- [模型文件（源码与权重）准备](#模型文件（源码与权重）准备)
    - [模型文件（源码与权重）下载及配置修改](#模型文件（源码与权重）下载及配置修改)
    - [视觉模型的 om 转换与其他的源码修改](#视觉模型的 om 转换与其他的源码修改)
- [模型推理](#模型推理)
- [测试](#测试)
    - [图像处理时间测试](#图像处理时间测试)
    - [精度测试](#精度测试)
    - [性能测试](#性能测试)
- [附录](#附录)
    - [视觉模型的 om 转换](#视觉模型的 om 转换)
    - [源码修改清单](#源码修改清单)

## 概述

MiniGPT-4 是兼具语言与图像理解能力的多模态模型，使用了先进的大语言模型强化了机器的视觉理解能力。具体来说，它结合了大语言模型 Vicuna 和视觉编码器 BLIP-2，具备强大的新型视觉语言能力。

- 参考实现：

  ```
   https://github.com/Vision-CAIR/MiniGPT-4
  ```

## 环境准备

该模型的软硬件依赖如下

**表 1** 硬件要求（任一）

| CPU     | Device |
|---------|--------|
| aarch64 | 310P   |
| aarch64 | 910B   |

**表 2** 推理引擎依赖

| 软件    | 版本要求      |
|-------|-----------|
| glibc | > = 2.27  |
| gcc   | > = 7.5.0 |

**表 3** Ascend 版本配套表

| 配套                 | 版本     | 下载链接 |
|--------------------|--------|------|
| Ascend HDK         | -      | -    |
| CANN               | -      | -    |
| python             | 3.9.18 | -    |
| FrameworkPTAdapter | 2.0.1  | -    |

### NPU 环境准备（HDK、CANN、PTA）

#### 安装 HDK

先安装 firmware，再安装 driver

##### 安装 firmware

安装方法（例）:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run |

```bash
# 安装 firmware（以 910b 为例）
chmod +x Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run
./Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run --full
```

##### 安装 driver

安装方法（例）：

| 包名                                                         |
|------------------------------------------------------------|
| Ascend-hdk-910b-npu-driver_23.0.rc3.b060_linux-aarch64.run |

```bash
# 安装 driver（以 arm、910b 为例）
chmod +x Ascend-hdk-910b-npu-driver_23.0.rc3.b060_linux-aarch64.run
./Ascend-hdk-910b-npu-driver_23.0.rc3.b060_linux-aarch64.run --full
```

#### 安装 CANN

先安装 toolkit，再安装 kernel

##### 安装 toolkit

安装方法（例）：

| 包名                                            |
|-----------------------------------------------|
| Ascend-cann-toolkit_7.0.T10_linux-aarch64.run |

```bash
# 安装 toolkit（以 arm 为例）
chmod +x Ascend-cann-toolkit_7.0.T10_linux-aarch64.run
./Ascend-cann-toolkit_7.0.T10_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

##### 安装 kernel

安装方法（例）：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-910b_7.0.T10_linux.run |

```bash
# 安装 kernel（以 910b 为例）
chmod +x Ascend-cann-kernels-910b_7.0.T10_linux.run
./Ascend-cann-kernels-910b_7.0.T10_linux.run --install
```

#### 安装 python 三方件（参见 requirements.txt）

```bash
pip install -r requirements.txt
```

此外，还需要安装 aclruntime 和 ais_bench 这两个三方件（为了支持 om 格式的模型）。请参考https:
//gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench 中的安装方式进行安装。

#### 安装 PytorchAdapter

先安装 torch，再安装 torch_npu

##### 安装 torch

安装方法（例）：

| 包名                                           |
|----------------------------------------------|
| torch-2.0.1-cp39-cp39-linux_aarch64.whl      |

```bash
# 安装 torch 2.0.1（以适配 arm、python 3.9 的版本为例）
pip install torch-2.0.1-cp39-cp39-linux_aarch64.whl
```

##### 安装 torch_npu

安装方法（例）：

| 包名                          |
|-----------------------------|
| pytorch_v2.0.1_py39.tar.gz  |

```bash
# 安装 torch_npu（以适配 python 3.9、torch 2.0.1 的版本为例）
tar -zxvf pytorch_v2.0.1_py39.tar.gz
pip install torch_npu-2.0.1.post1_20240222-cp39-cp39-linux_aarch64.whl
```

### 推理环境准备（加速库、模型库）

#### 路径变量解释

| 变量名         | 含义                                              |  
|-------------|-------------------------------------------------|
| llm_path    | 加速库及模型库下载后放置在此目录                                |
| work_space  | 主工作目录                                           |
| model_path  | 开源权重等必要材料放置在此目录                                 | 
| script_path | 与精度、性能测试有关的工作脚本放置在此目录                           |
| image_path  | 推理所需的图片放置在此目录（我们用的是`${work_space}/examples_v2`） |

#### 安装加速库

根据版本发布链接，安装加速库，将加速库下载至 `${llm_path}` 目录

| 加速库包名                                                 |
|-------------------------------------------------------|
| Ascend-cann-atb_{version}_cxx11abi0_linux-aarch64.run |
| Ascend-cann-atb_{version}_cxx11abi1_linux-aarch64.run |
| Ascend-cann-atb_{version}_cxx11abi1_linux-x86_64.run  |
| Ascend-cann-atb_{version}_cxx11abi0_linux-x86_64.run  |

具体使用 cxx11abi0 还是 cxx11abi1，可通过如下 python 命令查询

```python
import torch

torch.compiled_with_cxx11_abi()
```

若返回 True 则使用 cxx11abi1，否则相反。

```bash
# 安装atb 
chmod +x Ascend-cann-atb_*.run
./Ascend-cann-atb_*.run --install
source /usr/local/Ascend/atb/set_env.sh
```

#### 安装模型库

根据版本发布链接，下载模型库至 `${llm_path}` 目录

| 大模型包名                                                                     |
|---------------------------------------------------------------------------|
| Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi1.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi1.tar.gz |

具体使用 cxx11abi0 还是 cxx11abi1，判断方法与安装加速库时相同

 ```bash
 # 安装模型仓
 cd ${llm_path}
 tar -xzvf Ascend-cann-llm_*.tar.gz
 source set_env.sh
 ```

#### 安装模型库中的 atb_speed_sdk

打开下载好的模型库，进入`ModelLink\mindie_ref\mindie_llm\atb_models\pytorch\examples\atb_speed_sdk`目录，
执行`pip install .`。

## 模型文件（源码与权重）准备

### 模型文件（源码与权重）下载及配置修改

1. 下载 MiniGPT-4 的源码。

   下载地址：https://github.com/Vision-CAIR/MiniGPT-4 。

   下载完成后，得到目录 `/xx/xx/MiniGPT-4-main`，此即为主工作目录`${work_space}`。

2. 下载 MiniGPT-4 的权重`prerained_minigpt4_7b.pth`。

   下载地址：https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing ，

   下载完成后，保存到路径`${model_path}/pretrain/`下。

   须修改配置文件`${work_space}/eval_configs/minigpt4_eval.yaml`以声明此路径。

   参考：`ckpt: "${model_path}/pretrain/prerained_minigpt4_7b.pth"`。

3. 下载大语言模型 Vicuna-7b 的权重。

   下载地址：https://hf-mirror.com//Vision-CAIR/vicuna-7b/tree/main 。

   下载完成后，保存到路径`${model_path}/weights/`下。

   须修改配置文件`${work_space}/minigpt4/configs/models/minigpt4_vicuna0.yaml`以声明此路径。

   参考：`llama_model: "${model_path}/weights/"`。

4. 下载图像模型 VIT、Qformer 的权重 eva_vit_g.pth、blip2_pretrained_flant5xxl.pth，
   以及 Bert(bert-base-uncased) 的 Tokenizer。

   下载地址分别是：

   https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth ，

   https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth ，

   https://hf-mirror.com//bert-base-uncased 。

   下载完成后，保存到路径`${model_path}/othfiles/`下。

   此路径下所需的全部文件如下：

   ```bash
   eva_vit_g.pth
   blip2_pretrained_flant5xxl.pth
   bert-base-uncased
     config.json
     tokenizer_config.json
     vocab.txt
   ```

   须作相应的配置修改：

   `${work_space}/minigpt4/models/eva_vit.py`，

   ```python
   state_dict = torch.load("${model_path}/othfiles/eva_vit_g.pth", map_location="cpu")
   ```

   `${work_space}/minigpt4/models/minigpt4.py`

   ```python
   q_former_model = "${model_path}/othfiles/blip2_pretrained_flant5xxl.pth",
   ```

   ```python
   q_former_model = cfg.get("q_former_model", "${model_path}/othfiles/blip2_pretrained_flant5xxl.pth")",
   ```

   ```python
   encoder_config = BertConfig.from_pretrained("${model_path}/othfiles/bert-base-uncased")
   ```

   `${work_space}/minigpt4/models/eva_vit_model.py`

   ```python
   encoder_config = BertConfig.from_pretrained("${model_path}/othfiles/bert-base-uncased")
   ```

### 视觉模型的 om 转换与其他的源码修改

见[附录](#附录)。

## 模型推理

1. 将本项目的 models 目录下的`modeling_vicuna_ascend.py`、`image_encoder.py`拷贝到`${work_space}/minigpt4/models`目录下；

   将`run_predict.py`拷贝到`${work_space}`目录下。

5. 在`${work_space}`目录下，执行如下命令：

   ```bash
   python run_predict.py --cfg-path eval_configs/minigpt4_eval.yaml --image-path ${image_path}/office.jpg --npu-id ${npu-id}
   ```

## 测试

### 图像处理时间测试

将图像处理部分转换为OM模型后，图像处理时间约为0.018s；GPU图像处理时间约为1.185s

### 精度测试

#### 方案

我们采用的精度测试方案是这样的：使用同样的一组图片，分别在 GPU 和 NPU 上执行推理，得到两组图片描述。
再使用 open_clip 模型作为裁判，对两组结果分别进行评分，以判断优劣。

#### 实施

1. <a href="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main" style="color:blue">下载 open_clip 的权重
   open_clip_pytorch_model.bin</a>
2. 收集 GPU 和 NPU 的推理结果，整理成类似 `./precision/GPU_NPU_result_example.json` 的形式。
3. 执行脚本 `./precision/clip_score_minigpt4.py`，参考命令：
   ```bash
   python clip_score_minigpt4.py --device 0 --model_weights_path open_clip_pytorch_model.bin --image_info GPU_NPU_result_example.json
   ```
   若得分比值>1，则说明 NPU 上表现更优。

### 性能测试

#### 方案

我们使用 atb_speed_sdk 进行性能测试。

#### 实施

1. 先将 `./performance/modeling_vicuna_ascend_performance.py` 复制到 `${model_path}/weights/` 下。
   再修改 `${model_path}/weights/config.json`，新增或修改以下键值对：
   ```json
   "auto_map": {
    "AutoModelForCausalLM": "modeling_vicuna_ascend_performance.LlamaForCausalLM"
    }
   ```

2. 配置 `./performance/config.ini`（注意确保 `model_path=${model_path}/weights/`）

3. 执行脚本 `./performance/main.py`，参考命令：
   ```bash
   python main.py
   ```

### 性能测试（旧）

在功能运行正常的基础上，执行以下步骤进行性能测试。

#### 1. 替换transformer库中的utils文件

以transformers 4.30.2为例，需要替换transformer库的原生utils文件，来执行性能测试。

执行如下命令，找到需要被替换的文件所在的路径。

`python -c "import os;import transformers;print(os.path.join(os.path.dirname(transformers.__file__), 'generation'))"`

可以得到：/root/miniconda3/envs/cytest/lib/python3.8/site-packages/transformers/generation

进入上面得到的目录下做如下操作：

（1）将utils.py文件备份一份，命名为utils_ori.py

（2）将pytorch/examples/atb_speed_sdk/atb_speed/common/transformers_patch/4.30.2/utils_performance_test_npu_greedy.py拷贝到当前目录下，并重命名为utils.py

#### 2. 替换权重路径下的modeling文件

1. 将pytorch/examples/minigpt4/models/modeling_vicuna_ascend_performance.py拷贝到权重目录`${model_path}/weights/`下
2. 修改config.json , 将第5行 `"bos_token_id": 1,` 前的内容修改如下:

   ```json
   "auto_map": {
      "AutoModelForCausalLM":"modeling_vicuna_ascend_performance.LlamaForCausalLM"
   },
   ```

#### 3. 修改性能测试脚本内容

将`run_performance.py`的第188行修改为对应的输入输出长度：`temp = [[${input_length}, ${output_length}]]`

#### 4. 执行性能测试

 ```bash
  ATB_CONTEXT_TILING_RING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048" python run_performance.py --model_path ${model_path} --device_id ${device_id}] --batch ${batch_size}
 ```

为了不影响正常使用，将RETURN_PERF_DETAIL设置成1来返回具体的性能测试的值，默认是0

测试了batch_size为1，输入输出长度分别为[[256, 64], [512, 128], [1024, 256], [3584, 512]]
，输出 `multi_batch_performance.csv`

NPU性能测试结果如下：

| batch_size | input_length | output_length | reponse_time(ms) |   首token耗时（ms）   | 非首token平均耗时（ms）  | E2E吞吐（token/s） |
|:----------:|:------------:|:-------------:|:----------------:|:----------------:|:----------------:|:--------------:|
|     1      |     256      |      64       | 847.203969955444 | 5.98406791687011 | 13.3526968577551 |  75.54261107   |
|     1      |     512      |      128      | 1736.02199554443 | 11.7170810699462 | 13.5772040509802 |  73.73178469   |
|     1      |     1024     |      256      | 3599.91073608398 | 35.1405143737792 | 13.9794910655302 |  71.11287439   |
|     1      |     3584     |      512      | 8483.46471786499 | 344.723224639892 | 15.9270870708906 |  60.35269987   |

## 附录

### 视觉模型的 om 转换

MiniGPT-4 为多模态模型，其中图像处理部分的逻辑是固定的，比较适合转换为离线模型以提高性能

整个过程分为两步，第一步使用 `torch.onnx.export` 把需要转换的模型转成 onnx 格式，
第二步使用昇腾 ATC 工具将 onnx 模型转换为 om 模型。

#### onnx 转换

1. 首先，识别出图像处理部分的代码。即原始代码中`minigpt4.py`的第 125 行的 `image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)`
   及其配套代码。将这一部分单独写成一个文件，即为`eva_vit_model.py`。
   将它拷贝到`${work_space}/minigpt4/models`目录下。


2. 基于这部分模型代码，使用 `torch.onnx.export` 将相应的权重转换为 onnx 格式，详见 onnx_model_export.py。
   运行该文件，即可得到 onnx 模型。
   参考运行命令:
   ```bash
   python onnx_model_export.py --onnx-model-dir onnx模型的输出路径 --image-path 图片输入的路径（建议使用${image_path}中的图片）
   ```

#### om 转换

om 转换需使用昇腾 ATC 工具，参考
https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000005.html

1. 环境准备：安装 CANN 并 source 相应的环境变量；可参考上述链接中环境搭建的部分；

2. 模型转换：参考快速入门中 onnx 网络模型转换成离线模型章节，或参考执行下面的转换命令
   （要进入到已转换好的 onnx 模型目录中去执行上述命令，否则会找不到权重文件）：
   ```bash
   atc --model=eva_vit_g.onnx --framework=5 --output=${output_path}/eva_vit_g --soc_version=Ascend910B3 --input_shape="input:1,3,224,224"
   ```

### 源码修改清单

1. `${work_space}/minigpt4/models/base_model.py`文件，具体修改如下：

   （1）删除不必要的三方件引入（训练才需要）

   删除
   ```python
   from peft import (
       LoraConfig,
       get_peft_model,
       prepare_model_for_int8_training,
   )
   ```

   （2）改变 modeling 文件指向

   将
   ```python
   from minigpt4.models.modeling_llama import LlamaForCausalLM
   ```
   替换为
   ```python
   from minigpt4.models.modeling_vicuna_ascend import LlamaForCausalLM
   ```

2. `${work_space}/minigpt4/models/minigpt_base.py`文件，具体修改如下：

   （1）在文件开头导入图像 om 模型推理类

   ```python
   from minigpt4.models.image_encoder import IMAGE_ENCODER_OM
   ```

   （2）在第 40 行新增如下代码，初始化加载 om 模型

   ```python
   self.image_encoder = IMAGE_ENCODER_OM("${om_model_path}/", device_8bit)
   ```

   （3）删除原来的图像处理代码

   ```python
   self.visual_encoder, self.ln_vision = self.init_vision_encoder(
       vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit
   )
   ```

3. `${work_space}/minigpt4/models/minigpt4.py`文件，具体修改如下：

   （1）在文件开头导入 om 模型推理类

   ```python
   from ais_bench.infer.interface import InferSession
   ```

   （2）在原文件的第 63 行和 70 行，将`self.visual_encoder.num_features`修改为 VisionTransformer 类的入参 embed_dim 的固定值 1408.

   ```python
   self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408, freeze_qformer)
   ```

   ```python
   img_f_dim = 1408 * 4
   ```

   （3）修改原文件第 125 行，图像 embedding 的计算不再走原始逻辑，改用转换后的 om 模型进行计算

   ```python
   image_embeds = torch.tensor(self.image_encoder.image_encoder_om.infer(image.cpu().numpy())[0]).to(device)
   ```

4. `${work_space}/minigpt4/datasets/data_utils.py`文件，具体修改如下：

   （1）删除原文件第 18、19 行

   ```python
   import decord
   from decord import VideoReader
   ```

   （2）删除原文件第 29 行

   ```python
   decord.bridge.set_bridge("torch")
   ```

5. `${work_space}/eval_configs/minigpt4_eval.yaml`文件，具体修改如下：

   （1）由于无法使用 CUDA 的 8 位优化器，需将`low_resource`参数值设置为`False`。
