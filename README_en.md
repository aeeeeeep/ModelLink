<p align="center"> <img src="sources/images/logo.png" height="90px" width="400px"> </p>

<p align="center">
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
        <b><a href="README.md">简体中文</a> </b> |
        <b>English</b>
</p>

ModelLink aims to provide end-to-end large language model solutions for Huawei Ascend chips, including models, algorithms, and downstream tasks.

## ModelLink Solution Overview

---


### Supported features
Current ModelLink supported features for large model usage:
* [Dataset Preparation for Pre-training](#jump11)/[Fine-tuning Instruction Dataset Preparation](#jump12)
* [Pre-training](#jump13)/[Full-parameter Fine-tuning](#jump14)/[Low-parameter Fine-tuning](#jump15)
* [Inference: human-machine dialogue](#jump16)
* [Evaluation with numerous benchmarks](#jump17)
* [Utilizing Acceleration Features (Acceleration Algorithms + Fusion Operators)](#jump18)
* [Profiling data based on Ascend chips](#jump19)
* [Convert ckpt between huggingface and megatron](#jump19)
* [Enbale deterministic computing function for Ascend](#jump21)
* [Enable high availability features based on Ascend chips ](#jump22)

More novel and useful features are developing for LLMs training on Ascend ...

---

### Supporting version and models


【Please note the corresponding environment versions for model usage, as follows】

|           Software            | [Version](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          driver           |         under development version          |
|         firmware          |         under development version          |
|           CANN            |             under development version             |
|           torch           |            2.1.0、2.2.0             |
|         torch_npu         |           under development version           |


【Based on the current version of megatron, the performance statistics from our testing are as follows (Hardware info：Atlas 900 A2 PODc)】

For the supported following models, we provide training scripts and readme instructions in the examples folder, which contain detailed processes for model training, inference, and evaluation.

The following is the measured performance after turning on the mc2 acceleration feature, which is only supported in the version above 24RC2, the repository code level is closed by default, if you want to use, please refer to the "Acceleration Features" chapter on the home page.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Sequence length</th>
      <th>Cluster Scale</th>
      <th>Precision Mode</th>
      <th>Performance </th>
      <th>Reference Performance </th>
      <th>Contributor </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="examples/aquila/README.md">Aquila</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2849 </td>
      <td> 2874 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/aquila2/README_en.md">Aquila2</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 3323 </td>
      <td> 2673 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td> 4K </td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 854 </td>
      <td> 732 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan/README.md">Baichuan</a></td>
      <td>7B</td>
      <td> 4K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2685 </td>
      <td> 2036 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 4K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 1213 </td>
      <td> 862 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan2/README.md">Baichuan2</a></td>
      <td>7B</td>
      <td> 4K </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2664 </td>
      <td> 3969 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 4K </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1668 </td>
      <td> 2062 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td>7B1</td>
      <td> 2K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2034 </td>
      <td> 2525 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td >176B</td>
      <td> 2K </td>
      <td >12x8</td>
      <td> BF16 </td>
      <td> 100 </td>
      <td> 107 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/chatglm3/README.md">ChatGLM3</a></td>
      <td>6B</td>
      <td> 8K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 4297 </td>
      <td> 4267 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/codellama/README.md">CodeLlama</a></td>
      <td>34B</td>
      <td> 4K </td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 837 </td>
      <td> 762 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td>1x8</td>
      <td>BF16</td>
      <td> 2776 </td>
      <td> 2854 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td >65B</td>
      <td> 2K </td>
      <td >4x8</td>
      <td> BF16 </td>
      <td> 341 </td>
      <td> 414 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama/README.md">LLaMA</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 3600 </td>
      <td> 3804 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 2K </td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 1895 </td>
      <td> 2012 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
        <td>33B</td>
        <td> 2K </td>
        <td>4x8</td>
        <td>FP16</td>
        <td>621</td>
        <td>776</td>
        <td>【Ascend】</td>
    </tr>
    <tr>
      <td>65B</td>
      <td> 2K </td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 348 </td>
      <td> 426 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</a></td>
      <td>7B</td>
      <td> 4K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 4200 </td>
      <td> 3850 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 4K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1990 </td>
      <td> 1920 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td> 4K </td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 749 </td>
      <td> 796 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>70B</td>
      <td> 4K </td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 420 </td>
      <td> 430 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/llama3/README.md">LLaMA3</a></td>
      <td>8B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2483 </td>
      <td> 2674 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>70B</td>
      <td> 8K </td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 283 </td>
      <td> 355 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2499 </td>
      <td> 2867 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>14B</td>
      <td> 2K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1560 </td>
      <td> 1578 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>72B</td>
      <td> 8K </td>
      <td>16x8</td>
      <td>BF16 </td>
      <td> 285 </td>
      <td> 345 </td>
      <td>【Ascend】</td>
    </tr>
   <tr>
      <td rowspan="7"><a href="examples/qwen15/README.md">Qwen1.5</a></td>
      <td> 0.5B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 22834 </td>
      <td> 25306 </td>
      <td>【Community】</td>
      <tr>
      <td> 1.8B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 13029 </td>
      <td> 12181 </td>
      <td>【Community】</td>
      <tr>
      <td> 4B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  5033 </td>
      <td> 5328 </td>
      <td>【Community】</td>
      </tr>
      <td> 7B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  2862 </td>
      <td> 2621 </td>
      <td>【Community】</td>
      <tr>
      <td> 14B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1717 </td>
      <td> 1702 </td>
      <td>【Community】</td>
      <tr>
      <td> 32B </td>
      <td> 8K </td>
      <td> 4x8 </td>
      <td> BF16 </td>
      <td> 751 </td>
      <td> 708 </td>
      <td>【Community】</td>
      <tr>
      <td> 72B </td>
      <td> 8K </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> 301 </td>
      <td> 317 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/yi/README_en.md">Yi</a></td>
      <td>34B</td>
      <td> 4K </td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 809 </td>
      <td> 730 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td> 32K </td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 702 </td>
      <td> 837 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mistral/README.md">Mistral</a></td>
      <td>7B</td>
      <td> 32K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2806 </td>
      <td> 2734 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/gemma/README.md">Gemma</a></td>
      <td>2B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 6821 </td>
      <td> 7602 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2938 </td>
      <td> 2607 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/gpt3/README.md">GPT3</a></td>
      <td>175B</td>
      <td> 2K </td>
      <td> 16x8 </td>
      <td> FP16 </td>
      <td> 153 </td>
      <td> -- </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/README.md">Grok1</a></td>
      <td>40B</td>
      <td> 8K </td>
      <td> 2x8 </td>
      <td> BFP16 </td>
      <td> 1646 </td>
      <td> 2057 </td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>

---


### <span id="jump18">Acceleration Features
ModelLink supports various acceleration algorithms such as tensor parallelism, pipeline parallelism, context parallelism, sequence parallelism, recomputation, distributed optimizer, and more. The table below shows the enable switches corresponding to each acceleration feature:

<table><thead>
  <tr>
    <th>Scenario</th>
    <th>Features</th>
    <th>Arguments</th>
    <th>Mcore Support</th>
    <th>Legacy Support</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">PTD Parallel</td>
    <td>Tensor Parallel</td>
    <td>--tensor-model-parallel-size</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Pipeline Parallel</td>
    <td>--pipeline-model-parallel-size</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Dynamic division for PP</td>
    <td>--num-layer-list</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Sequence Parallel</td>
    <td>--sequence-parallel</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Distributed Optimizer</td>
    <td>--use-distributed-optimizer</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td rowspan="3">Context Parallel</td>
    <td>Context Parallel</td>
    <td>--context-parallel-size</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>Various Cp Algorithm</td>
    <td>--context-parallel-algo</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>Send/Recv Overlap</td>
    <td>--cp-send-recv-overlap</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td rowspan="2">MOE Parallel</td>
    <td>MOE Parallel</td>
    <td>--expert-model-parallel-size</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>MOE permutation communication optimization</td>
    <td>--moe-permutation-async-comm</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>Memory Optimization</td>
    <td>Re-computation</td>
    <td>--recompute-granularity</td>
    <td>No</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td rowspan="5">Fused Kernel</td>
    <td>Flash Attention</td>
    <td>--use-flash-attn</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Fused Rmsnorm</td>
    <td>--use-fused-rmsnorm</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Fused Swiglu</td>
    <td>--use-fused-swiglu</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Fused Rotary Position Embedding</td>
    <td>--use-fused-rotary-pos-emb</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Sliding Window Attention</td>
    <td>--sliding-window</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td rowspan="3">Communication</td>
    <td>Overlap Grad Reduce</td>
    <td>--overlap-grad-reduce</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Overlap Param Gather</td>
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
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layer-list 1,2,2,2,1 \
    --sequence-parallel \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 72 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-fused-rotary-pos-emb \
    --use-mc2 \
    --sliding-window 4096 \
    ... \
    ...
```
```bash
Note:
To enable mc2, ensure the following:
1. The environment version matches the description on the repository homepage;
2. Comment out line 431 in the validate_args_decorator function within modellink\arguments.py:
   #args.use_mc2 = False
```

---

## <span id="jump19"> Analyze profiling data based on Ascend chips 
Modellink supports analyze profiling data based on Ascend chips, which is useful for modelling:

```bash
--profile                        # enable profiling
--profile-step-start  5          # the start step
--profile-step-end 6             # the end step
--profile-ranks 0 1 2 3 4        # ranks for profiling
--profile-level level2           # level0, 1, 2 for data profiling
--profile-with-cpu               # profiling cpu information
--profile-with-stack             # profile stack information
--profile-with-memory            # profile memory information
--profile-record-shapes          # profile shape information
--profile-save-path ./profile_dir    # path to save data
```

## <span id="jump21"> Enable deterministic computing based on Ascend chips 

- add choice in script
```shell
--use-deter-comp
```
- add environment variable
```shell
export HCCL_DETERMINISTIC=True
```

---


## <span id="jump22"> Enable high availability features based on Ascend chips 
The motivation for the distributed optimizer is to save memory by distributing the optimizer state evenly across data parallel ranks.Based on this idea,a scheme is designed to divide the data parallel ranks into two replica data parallel ranks.The replica optimizer distributes the optimizer state evenly across replica data parallel ranks,so that the optimizer state is backed up. The following functions can be implemented based on the Huawei-developed HA framework:
1. During the training, the last checkpoint can be saved in the fault scenario, ensuring zero loss of the training result.
2. During the training, UCE fault detection of the HBM is supported, and online repair is completed to achieve step-level recomputation.

When the HA feature is enabled, the static memory used by the replica optimizer increases., the theoretical number of bytes per parameter is (where 'd' is the data parallel size):：

|                                  | Non-distributed optim | Distributed optim | Replica optim |
|----------------------------------| ------ | ------ |---------------|
| fp16/bf16 param, fp16/bf16 grads | 20 | 4 + 16/d | 4 + 32/d       |
| fp16/bf16 param, fp32 grads      | 18 | 6 + 12/d | Supporting      |
| fp32 param, fp32 grads           | 16 | 8 + 8/d  | Supporting      |


- add choice in script，and install Huawei-developed HA framework:mindio_ttp.whl
- mindio_ttp related description:https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/mindio/mindiottp
```shell
--enable-high-availability           #enable high availability feature
--enable-optimizer-state-local-copy  #enable the function of saving the last step optimizer state. Memory will increase further. It can be disabled by default.
```

---

## Acknowledgments

ModelLink is jointly contributed by the following departments of Huawei Corporation:
- Ascend Computing Product Unit
- Algorithm Unit of Computing Product Unit
- Research Unit of Computing Product Unit
- Open Computing Kit of Computing Product Unit
- General Development Department
- Global Technical Service Department

We appreciate every PR from community, and welcome to contribute to ModelLink.

## Appendix

- Safety Statement: [Safety Statement](https://gitee.com/ascend/ModelLink/wikis/%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)