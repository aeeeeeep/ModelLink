  <p align="center"> <img src="sources/images/logo.png" height="103px" width="700px"> </p>

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

ModelLink旨在为华为 [昇腾芯片](https://www.hiascend.com/) 上提供端到端的大语言模型方案, 包含模型，算法，以及下游任务。

---

## ModelLink大模型方案概览


当前ModelLink支撑大模型使用功能:
* [制作预训练数据集](#jump11)/[制作指令微调数据集](#jump12) 【NAIE】【昇腾】
* [预训练](#jump13)/[全参微调](#jump14)/[低参微调](#jump15) 【昇腾】【GTS】【NAIE】
* [流式推理/人机对话](#jump16) 【NAIE】【昇腾】
* [评估基线数据集](#jump17)【NAIE】
* [加速算法/融合算子/并行策略](#jump18)【昇腾】【计算算法部】【计算研究部】
* [基于昇腾芯片采集Profiling数据](#jump19) 【昇腾】
* [Huggingface与Megatron-LM权重转换](#jump20) 【昇腾】【OCK】
* [基于昇腾芯片的确定性计算功能](#jump21) 【昇腾】
* [基于昇腾芯片的高可用特性](#jump22) 【计算研究部】

强化学习等特性持续研发中....

---


## ModelLink版本维护策略

ModelLink版本有以下五个维护阶段：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划                | 1—3 个月 | 计划特性                                                                 |
| 开发                | 3 个月   | 开发特性                                                                 |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的ModelLink版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                                           |


ModelLink已发布版本维护策略：

| **ModelLink版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**         | **EOL日期** |
|-----------------|-----------|--------|------------|-----------------------|-----------|
| bk_origin_23    |  Demo     | EOL    | 2023       | 生命周期终止           | 2024/6/30 |
| 1.0             |  常规版本  | 维护   | 2024/03/30 | 预计2024/9/30起无维护  |           |
| 1.1             |  常规版本  | 维护   | 2024/06/30 | 预计2024/12/30起无维护 |           |

---

## 配套版本与支持模型

【版本配套环境】

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          Driver           |         在研版本          |
|         Firmware          |         在研版本          |
|           CANN            |             在研版本             |
|           Torch           |            2.1.0、2.2.0             |
|         Torch_npu         |           在研版本           |

【预训练集群性能与线性度】

ModelLink 通过模型并行与数据并行来训练大语言模型，为了演示如何使用多个昇腾芯片和模型大小进行扩展性训练，我们使用 `GPT3-175B` 稠密大模型，从128颗 NPU 扩展到 7968颗 NPU 进行实验，下图是实验数据：
<p align="center"> <img src="sources/images/linearity&mfu.png" height="485px" width="710px"> </p>
报告的吞吐量是针对端到端训练进行测量的，涵盖所有操作，包括数据加载、优化器步骤、通信，甚至日志记录。请注意，示例大模型没有训练至收敛。

图中呈现了对应集群规模下的 `MFU` 值与集群整体的 `线性度`情况. 计算公式已经放到社区，点击链接可进行参考：[MFU计算公式](https://gitee.com/ascend/ModelLink/wikis/%E6%9C%AF%E8%AF%AD%E5%AE%9A%E4%B9%89/%E5%A4%A7%E6%A8%A1%E5%9E%8B%20MFU%20%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F)，[线性度计算公式](https://gitee.com/ascend/ModelLink/wikis/%E6%9C%AF%E8%AF%AD%E5%AE%9A%E4%B9%89/%E7%BA%BF%E6%80%A7%E5%BA%A6%E5%85%AC%E5%BC%8F)

【现版本实测性能（硬件信息：Atlas 900 A2 PODc）】

下述列表中支持的模型，我们在[examples/README.md](./examples/README.md)中提供了相应的使用说明，里面有详细的模型训练、推理、评估流程

`参数`列中的超链接指向模型的预训练文件下载地址，`模型`列中的超链接指向更多的社区资源地址，包括Chat/Instruct权重等

`认证`【Pass】表示经过昇腾官方版本测试的模型，【Test】表示待测试模型

表中为开启 mc2 特性后预训练实测性能，该特性只在24RC2以上版本支持，本仓库代码层面默认关闭，若要使用，请参考[加速算法与融合算子](#加速算法与融合算子)章节

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
      <th>模式</th>
      <th>性能</th>
      <th>参考</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/collections/BAAI/aquila-6698657124de09d10cd7a83f">Aquila</a></td>
      <td><a href="https://huggingface.co/BAAI/Aquila-7B/tree/main">7B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2849 </td>
      <td> 2874 </td>
      <td><center>【GTS】</center></td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/collections/BAAI/aquila-6698657124de09d10cd7a83f">Aquila2</a></td>
      <td><a href="https://huggingface.co/BAAI/Aquila2-7B/tree/main">7B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 3323 </td>
      <td> 2673 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/BAAI/Aquila2-34B/tree/main">34B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 854 </td>
      <td> 732 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/baichuan-inc">Baichuan</a></td>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main">7B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2685 </td>
      <td> 2036 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main">13B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 1213 </td>
      <td> 862 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/baichuan-inc">Baichuan2</a></td>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main">7B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2664 </td>
      <td> 3969 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/tree/main">13B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1668 </td>
      <td> 2062 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/bigscience">Bloom</a></td>
      <td><a href="https://huggingface.co/bigscience/bloom-7b1/tree/main">7B1</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2034 </td>
      <td> 2525 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/bigscience/bloom/tree/main">176B</td>
      <td>2K</td>
      <th>Legacy</th>
      <td >12x8</td>
      <td> BF16 </td>
      <td> 100 </td>
      <td> 107 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/THUDM">ChatGLM3</a></td>
      <td><a href="https://huggingface.co/THUDM/chatglm3-6b/tree/main">6B</a></td>
      <td>8K</td>
      <th>Mcore</th>
      <td >1x8</td>
      <td> FP16 </td>
      <td> 4611 </td>
      <td> 4543 </td>
      <td>【昇腾】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/chatglm3-6b/tree/main">6B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td >1x8</td>
      <td> FP16 </td>
      <td> 2650 </td>
      <td> 2887 </td>
      <td>【昇腾】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/chatglm3-6b/tree/main">6B</a></td>
      <td>64K</td>
      <th>Mcore</th>
      <td >2x8</td>
      <td> FP16 </td>
      <td> 1724 </td>
      <td> 2097 </td>
      <td>【昇腾】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/codellama">CodeLlama</a></td>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-hf/tree/main">34B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 837 </td>
      <td> 762 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/internlm">InternLM</a></td>
      <td><a href="https://huggingface.co/internlm/internlm-7b/tree/main">7B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td>BF16</td>
      <td> 2776 </td>
      <td> 2854 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td >65B</td>
      <td>2K</td>
      <th>Legacy</th>
      <td >4x8</td>
      <td> BF16 </td>
      <td> 341 </td>
      <td> 414 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/meta-llama">LLaMA</td>
      <td><a href="https://huggingface.co/ruibin-wang/llama-7b-hf/tree/main">7B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td>FP16</td>
      <td> 3600 </td>
      <td> 3804 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/ruibin-wang/llama-13b-hf">13B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td>FP16</td>
      <td> 1895 </td>
      <td> 2012 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/pinkmanlove/llama-33b-hf/tree/main">33B</a></td>
        <td>2K</td>
        <th>Legacy</th>
        <td>4x8</td>
        <td>FP16</td>
        <td>621</td>
        <td>776</td>
        <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/pinkmanlove/llama-65b-hf">65B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 348 </td>
      <td> 426 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/meta-llama">LLaMA2</td>
      <td><a href="https://huggingface.co/daryl149/llama-2-7b-hf/tree/main">7B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 4672 </td>
      <td> 3850 </td>
      <td><center>【NAIE】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/NousResearch/Llama-2-13b-hf/tree/main">13B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2016 </td>
      <td> 1920 </td>
      <td><center>【NAIE】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/tree/main">34B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 749 </td>
      <td> 796 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/meta-llama/Llama-2-70b-hf">70B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 420 </td>
      <td> 430 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">LLaMA3</td>
      <td><a href="https://huggingface.co/unsloth/llama-3-8b/tree/main">8B</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2483 </td>
      <td> 2674 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/v2ray/Llama-3-70B/tree/main">70B</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 283 </td>
      <td> 355 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://modelscope.cn/organization/LLM-Research">LLaMA3.1</td>
      <td><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B">8B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2280 </td>
      <td> 2520 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B">8B</a></td>
      <td>128K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 1297 </td>
      <td> -- </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B">70B</a></td>
      <td>8K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 399 </td>
      <td> -- </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/Qwen">Qwen</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen-7B/tree/main">7B</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2499 </td>
      <td> 2867 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen-14B/tree/main">14B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1560 </td>
      <td> 1578 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen-72B/tree/main">72B</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>16x8</td>
      <td>BF16 </td>
      <td> 285 </td>
      <td> 345 </td>
      <td><center>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    </tr>
       <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B/tree/main">0.5B</a> </td>
      <td> 8K </td>
      <th>Legacy</th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 22834 </td>
      <td> 25306 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main">1.8B</a> </td>
      <td> 8K </td>
      <th>Legacy</th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 13029 </td>
      <td> 12181 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-4B/tree/main">4B</a> </td>
      <td> 8K </td>
      <th>Legacy</th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 5033 </td>
      <td> 5328 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-7B/tree/main">7B</a> </td>
      <td> 8K </td>
      <th>Legacy</th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2862 </td>
      <td> 2621 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-14B/tree/main">14B</a> </td>
      <td> 8K </td>
      <th>Legacy</th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1717 </td>
      <td> 1702 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-32B/tree/main">32B</a> </td>
      <td> 8K </td>
      <th>Legacy</th>
      <td> 4x8 </td>
      <td> BF16 </td>
      <td> 751 </td>
      <td> 708 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-72B/tree/main">72B</a> </td>
      <td> 8K </td>
      <th>Legacy</th>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> 301 </td>
      <td> 317 </td>
      <td><center>【GTS】</td>    
      <td>【Pass】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-110B/tree/main">110B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> 223 </td>
      <td> -- </td>
      <td><center>【GTS】</td>    
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="7"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td rowspan="2"> <a href="https://huggingface.co/Qwen/Qwen2-0.5B/tree/main">0.5B</a> </td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 28618 </td>
      <td> 34859 </td>
      <td><center>【GTS】</td>
      <td><center>【Test】</td>
      <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 11338 </td>
      <td><center> -- </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td rowspan="2"> <a href="https://huggingface.co/Qwen/Qwen2-1.5B/tree/main">1.5B</a> </td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 15456 </td>
      <td> 15603 </td>
      <td><center>【GTS】</td>
      <td><center>【Test】</td>
      <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 7281 </td>
      <td><center> 8538 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2-7B/tree/main">7B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 4034 </td>
      <td>4241 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td> 32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2040 </td>
      <td>2045 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
      <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-72B/tree/main">72B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 368 </td>
      <td><center>-- </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr> 
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/01-ai">Yi</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-34B/tree/main">34B</a></td>
      <td> 4K</td>
      <th>Legacy</th>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 768 </td>
      <td> 730 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/mistralai">Mixtral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">8x7B</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 706 </td>
      <td> 837 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/mistralai">Mistral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main">7B</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2900 </td>
      <td> 2734 </td>
      <td><center>【NAIE】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma</a></td>
      <td><a href="https://huggingface.co/google/gemma-2b/tree/main">2B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 7067 </td>
      <td> 7602 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-7b">7B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2939 </td>
      <td> 2607 </td>
      <td><center>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2">GPT3</td>
      <td>175B</td>
      <td> 2K </td>
      <th>Legacy</th>
      <td> 16x8 </td>
      <td> FP16 </td>
      <td> 153 </td>
      <td> -- </td>
      <td><center>【昇腾】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td>15B</td>
      <td> 2K </td>
      <th>Legacy</th>
      <td> 1x8 </td>
      <td> FP16 </td>
      <td> 1890 </td>
      <td> 1840 </td>
      <td><center>【昇腾】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/xai-org/grok-1">Grok1</a></td>
      <td><a href="https://github.com/xai-org/grok-1">8x5B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 2x8 </td>
      <td> BF16 </td>
      <td> 1646 </td>
      <td> 2057 </td>
      <td><center>【昇腾】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a herf="https://github.com/OpenBMB/MiniCPM">MiniCPM</td>
      <td><a herf="https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/tree/main">2B</td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 7314 </td>
      <td> 7953 </td>
      <td><center>【NAIE】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/openbmb/MiniCPM-MoE-8x2B/tree/main">8x2B</td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2981 </td>
      <td> 3172 </td>
      <td><center>【NAIE】</td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>

---

## Huggingface与Megatron-LM权重转换

ModelLink支持Huggingface、Megatron-Legacy以及Megatron-Core之间的权重格式互转，具体功能列表如下：


<table>
  <thead>
    <tr>
      <th>源格式</th>
      <th>目标格式</th>
      <th>支持特性</th>
      <th>特性入参</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">HuggingFace </td>
      <td rowspan="4">Megatron-Legacy</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td rowspan="8">Megatron-Core</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--expert-model-parallel-size</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="20">Megatron-Legacy </td>
      <td rowspan="8">Huggingface</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td>LoRA训练模块</td>
      <td>--lora-target-modules</td>
    </tr>
    <tr>
      <td>LoRA权重</td>
      <td>--lora-load</td>
    </tr>
    <tr>
      <td>LoRA r</td>
      <td>--lora-r</td>
    </tr>
    <tr>
      <td>LoRA alpa</td>
      <td>--lora-alpha</td>
    </tr>
    <tr>
      <td rowspan="4">Megatron-Core</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td rowspan="6">Megatron-Legacy</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>LoRA训练模块</td>
      <td>--lora-target-modules</td>
    </tr>
    <tr>
      <td>LoRA权重</td>
      <td>--lora-load</td>
    </tr>
    <tr>
      <td>LoRA r</td>
      <td>--lora-r</td>
    </tr>
    <tr>
      <td>LoRA alpa</td>
      <td>--lora-alpha</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="13">Megatron-Core </td>
      <td rowspan="4">Huggingface</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td rowspan="4">Megatron-Legacy</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td rowspan="5">Megatron-Core</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--expert-model-parallel-size</td>
    </tr>
  </tbody>
</table>

具体的权重转换功能命令介绍见[examples/README.md](./examples/README.md)

---

## 预训练加速算法与融合算子

ModelLink预训练支持张量并行、流水线并行等多种加速算法和融合算子，下表为各种加速特性对应的使能开关：

<table><thead>
  <tr>
    <th>使用场景</th>
    <th>特性名称</th>
    <th>具体参数</th>
    <th>Mcore</th>
    <th>Legacy</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="6">PTD并行</td>
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
    <td>虚拟流水并行</td>
    <td>--num-layers-per-virtual-pipeline-stage</td>
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
    <td>Send/recv掩盖加速</td>
    <td>--cp-send-recv-overlap</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td rowspan="2">MOE</td>
    <td>MOE专家并行</td>
    <td>--expert-model-parallel-size</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>MOE重排通信优化</td>
    <td>--moe-permutation-async-comm</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td rowspan="6">显存优化</td>
    <td>参数副本复用</td>
    <td>--reuse-fp32-param</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>激活函数重计算</td>
    <td>--recompute-activation-function</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>重计算程度</td>
    <td>--recompute-granularity</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>重计算层数</td>
    <td>--recompute-num-layers</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>重计算方法</td>
    <td>--recompute-method</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>PP-Stage重计算</td>
    <td>--enable-recompute-layers-per-pp-rank</td>
    <td>Yes</td>
    <td>Yes</td>
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
    <td>Yes</td>
    <td>Yes</td>
  </tr>
</tbody></table>

```bash
注意：
如果需要开启 mc2，需保证:
1. 配套环境版本如本仓首页所述;
2. 将 modellink\arguments.py 文件下 validate_args_decorator 函数中的此处注释开启
   #args.use_mc2 = False
```

具体的预训练方法见[examples/README.md](./examples/README.md)

---

## 基于昇腾芯片采集Profiling数据
Modellink支持基于昇腾芯片采集profiling数据，以提供对模型运行情况的分析，主要API如下：


```bash
--profile                        # 打开profiling采集数据开关
--profile-step-start  5          # 指定开启采集数据的步骤
--profile-step-end 6             # 指定结束采集数据的步骤，实际采集步数为 end-start，不包含end
--profile-ranks 0 1 2 3 4        # 指定采集数据的卡号，默认为-1，表示采集所有rank的profiling数据，可以设置为 0 1 2 3 4 5 6 7 8 9 列表指定全局卡号
--profile-level level2           # 数据采集水平，level0, 1, 2, 级别越高采集信息越多，默认为level0
--profile-with-cpu               # 是否采集CPU数据，加入参数采集
--profile-with-stack             # 采集指令运行堆栈，加入参数采集
--profile-with-memory            # 是否采集内存，加入参数采集
--profile-record-shapes          # 是否采集计算shape，加入参数采集
--profile-save-path ./profile_dir    # profiling数据采集保存路径
```

---

## 基于昇腾芯片的确定性计算功能
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


## 基于昇腾芯片的高可用特性
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

---

## 致谢

ModelLink由华为公司的下列部门联合贡献 ：
- 昇腾计算产品部
- 计算算法部
- 计算研究部
- 开源计算工具部: OCK
- 公共开发部：NAIE
- 全球技术服务部：GTS

感谢来自社区的每一个PR，欢迎贡献 ModelLink

---

## 安全声明

[ModelLink安全声明](https://gitee.com/ascend/ModelLink/wikis/%E5%AE%89%E5%85%A8%E7%9B%B8%E5%85%B3/%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)
