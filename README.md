  <p align="center"> <img src="sources/images/logo.png" height="90px" width="400px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>
<p align="center">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b>
</p>

ModelLink旨在为华为 [昇腾芯片](https://open.codehub.huawei.com/OpenBaize/Ascend/ascendspeed/files?ref=master&filePath=examples%2Fbaichuan%2Fpretrain_baichuan_zero_7B.sh&isFile=true) 上提供端到端的大语言模型方案, 包含模型，算法，以及下游任务。

## ModelLink解决方案概览

---


### 支持功能
当前ModelLink支撑大模型使用功能:
* [制作预训练数据集](#jump11)/[制作指令微调数据集](#jump12)
* [预训练](#jump13)/[全参微调](#jump14)/[低参微调](#jump15)
* [流式推理/人机对话](#jump16)
* [评估基线数据集](#jump17)
* [加速算法/融合算子/并行策略](#jump18)
* [基于昇腾芯片采集Profiling数据](#jump19)
* [Huggingface与Megatron-LM权重转换](#jump20)
* [基于昇腾芯片的确定性计算功能](#jump21)
* [基于昇腾芯片的高可用特性](#jump22)

强化学习等特性持续研发中....

---

## 配套版本与支持模型

【需要注意模型使用时的配套环境版本，参考如下】

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          driver           |         在研版本          |
|         firmware          |         在研版本          |
|           CANN            |             在研版本             |
|           torch           |            2.1.0、2.2.0             |
|         torch_npu         |           在研版本           |


【基于现版本我们实测的性能情况统计如下（硬件信息：Atlas 900 A2 PODc）】

下述列表中支持的模型，我们在[examples](./examples/)文件夹中提供了相应的训练脚本和readme说明，里面有详细的模型训练、推理、评估流程。

以下为开启 mc2 加速特性后的实测性能，该特性只在24RC2以上版本支持，本仓库代码层面默认关闭，若要使用，请参考首页`加速算法与融合算子`章节

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>序列</th>
      <th>集群规模</th>
      <th>精度模式</th>
      <th>性能</th>
      <th>参考性能</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="examples/aquila/README.md">Aquila</a></td>
      <td>7B</td>
      <td>2K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2849 </td>
      <td> 2874 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/aquila2/README.md">Aquila2</a></td>
      <td>7B</td>
      <td>2K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 3323 </td>
      <td> 2673 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td>4K</td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 854 </td>
      <td> 732 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan/README.md">Baichuan</a></td>
      <td>7B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2685 </td>
      <td> 2036 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 1213 </td>
      <td> 862 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan2/README.md">Baichuan2</a></td>
      <td>7B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2664 </td>
      <td> 3969 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1668 </td>
      <td> 2062 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td>7B1</td>
      <td>2K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2034 </td>
      <td> 2525 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>176B</td>
      <td>2K</td>
      <td >12x8</td>
      <td> BF16 </td>
      <td> 100 </td>
      <td> 107 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/chatglm3/README.md">ChatGLM3</a></td>
      <td>6B</td>
      <td> 8K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 4297 </td>
      <td> 4267 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/codellama/README.md">CodeLlama</a></td>
      <td>34B</td>
      <td>4K</td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 837 </td>
      <td> 762 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>BF16</td>
      <td> 2776 </td>
      <td> 2854 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td >65B</td>
      <td>2K</td>
      <td >4x8</td>
      <td> BF16 </td>
      <td> 341 </td>
      <td> 414 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama/README.md">LLaMA</td>
      <td>7B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 3600 </td>
      <td> 3804 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 1895 </td>
      <td> 2012 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
        <td>33B</td>
        <td>2K</td>
        <td>4x8</td>
        <td>FP16</td>
        <td>621</td>
        <td>776</td>
        <td>【昇腾】</td>
    </tr>
    <tr>
      <td>65B</td>
      <td>2K</td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 348 </td>
      <td> 426 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</td>
      <td>7B</td>
      <td>4K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 4200 </td>
      <td> 3850 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1990 </td>
      <td> 1920 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td>4K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 749 </td>
      <td> 796 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>70B</td>
      <td>4K</td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 420 </td>
      <td> 430 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/llama3/README.md">LLaMA3</td>
      <td>8B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2483 </td>
      <td> 2674 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>70B</td>
      <td> 8K </td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 283 </td>
      <td> 355 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2499 </td>
      <td> 2867 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>14B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1560 </td>
      <td> 1578 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>72B</td>
      <td> 8K </td>
      <td>16x8</td>
      <td>BF16 </td>
      <td> 285 </td>
      <td> 345 </td>
      <td>【昇腾】</td>
    </tr>
    </tr>
       <tr>
      <td rowspan="7"><a href="examples/qwen15/README.md">Qwen1.5</a></td>
      <td> 0.5B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 22834 </td>
      <td> 25306 </td>
      <td>【社区】</td>
      <tr>
      <td> 1.8B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 13029 </td>
      <td> 12181 </td>
      <td>【社区】</td>
      <tr>
      <td> 4B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 5033 </td>
      <td> 5328 </td>
      <td>【社区】</td>
      <tr>
      <td> 7B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2862 </td>
      <td> 2621 </td>
      <td>【社区】</td>
      <tr>
      <td> 14B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1717 </td>
      <td> 1702 </td>
      <td>【社区】</td>
      <tr>
      <td> 32B </td>
      <td> 8K </td>
      <td> 4x8 </td>
      <td> BF16 </td>
      <td> 751 </td>
      <td> 708 </td>
      <td>【社区】</td>
      <tr>
      <td> 72B </td>
      <td> 8K </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> 301 </td>
      <td> 317 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/yi/README.md">Yi</a></td>
      <td>34B</td>
      <td> 4K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 809 </td>
      <td> 730 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td> 32K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 487 </td>
      <td> 610 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mistral/README.md">Mistral</a></td>
      <td>7B</td>
      <td> 32K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2806 </td>
      <td> 2734 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/gemma/README.md">Gemma</a></td>
      <td>2B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 6821 </td>
      <td> 7602 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2938 </td>
      <td> 2607 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/gpt3/README.md">GPT3</a></td>
      <td>175B</td>
      <td> 2K </td>
      <td> 16x8 </td>
      <td> FP16 </td>
      <td> 153 </td>
      <td> -- </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/README.md">Grok1</a></td>
      <td>40B</td>
      <td> 8K </td>
      <td> 2x8 </td>
      <td> BFP16 </td>
      <td> 1646 </td>
      <td> 2057 </td>
      <td>【昇腾】</td>
    </tr>
  </tbody>
</table>

---

## <span id="jump18"> 加速算法与融合算子

ModelLink支持张量并行、流水线并行、context并行、序列并行、重计算、分布式优化器等多种加速算法和融合算子，下表为各种加速特性对应的使能开关：

<table><thead>
  <tr>
    <th>使用场景</th>
    <th>特性名称</th>
    <th>具体参数</th>
    <th>Mcore支持</th>
    <th>Legacy支持</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">PTD并行</td>
    <td>张量并行</td>
    <td>--tensor-model-parallel-size</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>流水线并行</td>
    <td>--pipeline-model-parallel-size</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>流水线并行动态划分</td>
    <td>--num-layer-list</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>序列并行</td>
    <td>--sequence-parallel</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>分布式优化器</td>
    <td>--use-distributed-optimizer</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td rowspan="3">长序列并行</td>
    <td>长序列并行</td>
    <td>--context-parallel-size</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>多并行方案</td>
    <td>--context-parallel-algo</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>send/recv掩盖加速</td>
    <td>--cp-send-recv-overlap</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td rowspan="5">融合算子</td>
    <td>Flash attention</td>
    <td>--use-flash-attn</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Fused rmsnorm</td>
    <td>--use-fused-rmsnorm</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Fused swiglu</td>
    <td>--use-fused-swiglu</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Fused rotary position embedding</td>
    <td>--use-fused-rotary-pos-emb</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Sliding window attention</td>
    <td>--sliding-window</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td rowspan="3">通信</td>
    <td>梯度reduce通算掩盖</td>
    <td>--overlap-grad-reduce</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>权重all-gather通算掩盖</td>
    <td>--overlap-param-gather</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>MC2</td>
    <td>--use-mc2</td>
    <td>No</td>
    <td>Yes</td>
  </tr>
</tbody></table>



```bash
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layer-list 5,6,6,6,6,5 \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_ALGO} \
    --ulysses-degree-in-cp 2 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-fused-rotary-pos-emb \
    --use-mc2 \
    ... \
    ...
```
```bash
注意：
如果需要开启 mc2，需保证:
1. 配套环境版本如本仓首页所述;
2. 将 modellink\arguments.py 中 validate_args_decorator 函数中的第431行进行注释
   #args.use_mc2 = False
```
---

## <span id="jump19"> 基于昇腾芯片采集Profiling数据
Modellink支持基于昇腾芯片采集profiling数据，以提供对模型运行情况的分析，主要API如下：


```bash
--profile                        # 打开profiling采集数据开关
--profile-step-start  5          # 指定开启采集数据的步骤
--profile-step-end 6             # 指定结束采集数据的步骤，实际采集步数为 end-start，不包含end
--profile-ranks 0 1 2 3 4        # 指定采集数据的卡号，默认为0，可以设置为 0 1 2 3 4 5 6 7 8 9 列表指定全局卡号
--profile-level level2           # 数据采集水平，level0, 1, 2, 级别越高采集信息越多，默认为level0
--profile-with-cpu               # 是否采集CPU数据，加入参数采集
--profile-with-stack             # 采集指令运行堆栈，加入参数采集
--profile-with-memory            # 是否采集内存，加入参数采集
--profile-record-shapes          # 是否采集计算shape，加入参数采集
--profile-save-path ./profile_dir    # profiling数据采集保存路径
```

---

## <span id="jump21"> 基于昇腾芯片的确定性计算功能
昇腾芯片默认采用了不确定计算加速模型训练，有时为了重复实验与对比实验需要确定性的计算结果，ModelLink使能确定性计算的开关如下：

- 启动命令中加入开关
```shell
--use-deter-comp
```
- 环境变量中加入开关
```shell
export HCCL_DETERMINISTIC=True
```

---


## <span id="jump22"> 基于昇腾芯片的高可用特性
分布式优化器的思想是通过将优化器状态均匀地分布在数据并行组中来节省内存。基于该思想，设计了将数据并行组切分成两个副本数据并行组的方案，副本优化器将优化器状态均匀分布在副本数据并行组，实现优化器状态均有备份。结合华为自研的高可用框架，可实现以下功能：
1. 训练过程中，支持故障场景保存临终checkpoint，训练结果0损失。
2. 训练过程中，支持HBM的UCE故障检测，并完成在线修复，达到Step级重计算。

开启高可用特性时，副本优化器使用的静态内存有所增加，每个参数的理论字节数为（其中“d”是数据并行大小）：

|                                  | Non-distributed optim | Distributed optim | Replica optim |
|----------------------------------| ------ | ------ |---------------|
| fp16/bf16 param, fp16/bf16 grads | 20 | 4 + 16/d | 4 + 32/d       |
| fp16/bf16 param, fp32 grads      | 18 | 6 + 12/d | Supporting      |
| fp32 param, fp32 grads           | 16 | 8 + 8/d  | Supporting      |


- 启动命令中加入开关，并安装华为自研高可用框架mindio_ttp.whl
- mindio_ttp相关说明：https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/mindio/mindiottp
```shell
--enable-high-availability           #使能高可用特性的总开关
--enable-optimizer-state-local-copy  #使能保存上一步优化器状态，内存会进一步增加，默认可关闭
```


## 致谢

---

ModelLink由华为公司的下列部门联合贡献 ：
- 昇腾计算产品部
- 计算算法部
- 计算研究部
- 开源计算工具部
- 公共开发部
- 全球技术服务部

感谢来自社区的每一个PR，欢迎贡献 ModelLink

---

## 安全声明

[ModelLink安全声明](https://gitee.com/ascend/ModelLink/wikis/%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)