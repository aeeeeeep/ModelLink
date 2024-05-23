<p align="center"> <img src="sources/images/logo.png" height="90px" width="400px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/AscendSpeed/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://gitee.com/ascend/AscendSpeed">
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

ModelLink provides end-to-end solutions for large language models on Ascend chips, including models, algorithms and tasks.

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
* [convert ckpt between huggingface and megatron](#jump19)

More novel and useful features are developing for LLMs training on Ascend ...


### Supported Models
Current ModelLink supports pre-training and fine-tuning for the following models:
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Scale</th>
      <th>Pretrain</th>
      <th>Inference</th>
      <th>LoRA</th>
      <th>SFT</th>
      <th>Chat</th>
      <th>Evaluation</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="examples/aquila/README.md">Aquila</a> </td>
      <td> 7B </td>
      <td> <a href="examples/aquila/pretrain_aquila_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/aquila/generate_aquila_7b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/aquila/evaluate_aquila_7b_ptd.sh">eval</a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/aquila2/README.md">Aquila2</a></td>
      <td>7B</td>
      <td> <a href="examples/aquila2/pretrain_aquila2_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/aquila2/generate_aquila2_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/aquila2/evaluate_aquila2_7b_ptd.sh">eval</a> </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="examples/baichuan/README.md">Baichuan</a> </td>
      <td> 7B </td>
      <td> <a href="examples/baichuan/pretrain_baichuan_ptd_7B.sh"> pretrain </a> </td>
      <td> <a href="examples/baichuan/generate_baichuan_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/baichuan/evaluate_baichuan_7B_ptd.sh">eval</a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> <a href="examples/baichuan/pretrain_baichuan_ptd_13B.sh"> pretrain </a> </td>
      <td> <a href="examples/baichuan/generate_baichuan_13b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/baichuan/evaluate_baichuan_13B_ptd.sh"> eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="examples/baichuan2/README.md">Baichuan2</a> </td>
      <td> 7B </td>
      <td> <a href="examples/baichuan2/pretrain_baichuan2_ptd_7B.sh"> pretrain </a> </td>
      <td> <a href="examples/baichuan2/generate_baichuan2_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/baichuan2/evaluate_baichuan2_7B_ptd.sh">eval</a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> <a href="examples/baichuan2/pretrain_baichuan2_ptd_13B.sh"> pretrain </a> </td>
      <td> <a href="examples/baichuan2/generate_baichuan2_13b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/baichuan2/evaluate_baichuan2_13B_ptd.sh">eval</a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td> 7B1 </td>
      <td> <a href="examples/bloom/pretrain_bloom_ptd_7B.sh"> pretrain </a> </td>
      <td> <a href="examples/bloom/generate_bloom_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/bloom/evaluate_bloom_7B_ptd.sh">eval</a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td> 176B </td>
      <td> <a href="examples/bloom/pretrain_bloom_176b.sh"> pretrain </a> </td>
      <td> <a href="examples/bloom/generate_bloom_176b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/bloom/evaluate_bloom_176b_ptd.sh">eval</a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="1"> <a href="examples/codellama/README.md">CodeLlama</a> </td>
      <td> 34B </td>
      <td> <a href="examples/codellama/pretrain_codellama_34b_ptd_16p.sh"> pretrain </a> </td>
      <td> <a href="examples/codellama/generate_codellama_34b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/codellama/evaluate_codellama_34b_ptd.sh">eval</a> </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td> <a href="examples/intern/pretrain_internlm_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/intern/generate_internlm_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/intern/evaluate_internlm_7B_ptd.sh">eval</a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td>65B</td>
      <td> <a href="examples/intern/pretrain_internlm_65b_ptd.sh"> pretrain </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama/README.md">LLaMA</a></td>
      <td>7B</td>
      <td> <a href="examples/llama/pretrain_llama_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama/generate_llama_7b_ptd.sh"> generate </a> </td>
      <td> <a href="examples/llama/tune_llama_7b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama/evaluate_llama_7B_ptd.sh"> eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td> <a href="examples/llama/pretrain_llama_13b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama/generate_llama_13b_ptd.sh">generate </a> </td>
      <td> <a href="examples/llama/tune_llama_13b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama/evaluate_llama_13B_ptd.sh">eval</a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td>33B</td>
      <td> <a href="examples/llama/pretrain_llama_33B_ptd_32p.sh"> pretrain </a> </td>
      <td> <a href="examples/llama/generate_llama_33b_ptd.sh">generate </a> </td>
      <td> <a href="examples/llama/tune_llama_33b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama/evaluate_llama_33B_ptd.sh">eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td > 65B </td>
      <td> <a href="examples/llama/pretrain_llama_65b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama/generate_llama_65b_ptd.sh"> generate </a> </td>
      <td > <a href="examples/llama/tune_llama_65b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama/evaluate_llama_65B_ptd.sh">eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</a></td>
      <td>7B</td>
      <td> <a href="examples/llama2/pretrain_llama2_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama2/generate_llama2_7b_ptd.sh">generate </a> </td>
      <td> <a href="examples/llama2/tune_llama2_7b_ptd.sh">lora </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama2/evaluate_llama2_7B_ptd.sh">eval </a>  </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td> <a href="examples/llama2/pretrain_llama2_13B_ptd_8p.sh"> pretrain </a> </td>
      <td> <a href="examples/llama2/generate_llama2_13b_ptd.sh">generate </a> </td>
      <td> <a href="examples/llama2/tune_llama2_13b_ptd.sh">lora </a> </td>      
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama2/evaluate_llama2_13B_ptd.sh">eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td> 34B </td>
      <td> <a href="examples/llama2/pretrain_llama2_34B_ptd_16p.sh"> pretrain </a> </td>
      <td> <a href="examples/llama2/generate_llama2_34B_ptd.sh"> generate </a> </td>
      <td> <a href="examples/llama2/tune_llama2_34b_ptd.sh">lora </a> </td>
      <td > -- </td>
      <td > -- </td>
      <td> <a href="examples/llama2/evaluate_llama2_34B_ptd.sh">eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td > 70B </td>
      <td> <a href="examples/llama2/pretrain_llama2_70b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama2/generate_llama2_70b_ptd.sh"> generate </a> </td>
      <td > <a href="examples/llama2/tune_llama2_70b_ptd.sh">lora</a> </td>
      <td > -- </td>
      <td > -- </td>
      <td> <a href="examples/llama2/evaluate_llama2_70B_ptd.sh">eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/llama3/README.md">LLaMA3</a></td>
      <td>8B</td>
      <td> <a href="examples/llama3/pretrain_llama3_8b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama3/generate_llama3_8b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama3/generate_llama3_8b_chat_ptd.sh"> chat </a> </td>
      <td> <a href="examples/llama3/evaluate_llama3_8b_ptd.sh"> eval </a>  </td>
      <td> 【Community】 </td>
    </tr>
    <tr>
      <td>70B</td>
      <td> <a href="examples/llama3/pretrain_llama3_70b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama3/generate_llama3_70b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama3/evaluate_llama3_70b_ptd.sh"> eval </a> </td>
      <td> 【Community】 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td> <a href="examples/qwen/pretrain_qwen_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen/generate_qwen_7b_ptd.sh"> generate </a></td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen/evaluate_qwen_7b_ptd.sh">eval </a></td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td>14B</td>
      <td> <a href="examples/qwen/pretrain_qwen_14b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen/generate_qwen_14b_ptd.sh">generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen/evaluate_qwen_14b_ptd.sh"> eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td>72B</td>
      <td> <a href="examples/qwen/pretrain_qwen_72b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen/generate_qwen_72b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen/evaluate_qwen_72b_ptd.sh"> eval </a> </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/qwen15/README.md">Qwen1.5</a></td>
      <td>7B</td>
      <td> <a href="examples/qwen15/pretrain_qwen15_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_7b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_7b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【Community】 </td>
    </tr>
      <td>14B</td>
      <td> <a href="examples/qwen15/pretrain_qwen15_14b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_14b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_14b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【Community】 </td>
    <tr>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/yi/README.md">Yi</a></td>
      <td>34B</td>
      <td> <a href="examples/yi/pretrain_yi_34b_ptd_16p.sh"> pretrain </a> </td>
      <td> <a href="examples/yi/generate_yi_34b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/yi/evaluate_yi_34b_ptd.sh"> eval </a> </td>
      <td> 【Community】 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td> <a href="examples/mixtral/pretrain_mixtral_8x7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/mixtral/generate_mixtral_8x7b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/mixtral/evaluate_mixtral_8x7b_ptd.sh"> eval </a>  </td>
      <td> 【Ascend】 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mistral/README.md">Mistral</a></td>
      <td>7B</td>
      <td> <a href="examples/mistral/pretrain_mistral_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/mistral/generate_mistral_7b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/mistral/evaluate_mistral_7b_ptd.sh"> eval </a>  </td>
      <td>【Ascend】</td>
    <tr>
      <td rowspan="1"><a href="examples/gemma/README.md">Gemma</a></td>
      <td>7B</td>
      <td> <a href="examples/gemma/pretrain_gemma_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/gemma/generate_gemma_7b_ptd.sh"> generate </a></td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/gemma/evaluate_gemma_7b_ptd.sh">eval </a></td>
      <td> 【Ascend】 </td>
    </tr>
  </tbody>
</table>


### Script Naming Rules
|      Script       |        Rule         |
|:-----------------:|:-------------------:|
|  pretrain_xxx.sh  | Pre-training Script |
|    tune_xxx.sh    | Fine-tuning Script  |
|  generate_xxx.sh  |  Inference Script   |
|  xxx_chat_xxx.sh  |     Chat Script     |
| evaluation_xxx.sh |  Evaluation Script  |

---

# Model Usage Guide and Version Notes


Model Usage Guide and Version Notes
For the supported models listed above, we provide training scripts and readme instructions in the examples folder, which contain detailed processes for model training, inference, and evaluation.

【Please note the corresponding environment versions for model usage, as follows】

|           Software            | [Version](https://www.hiascend.com/zh/) |
| :-----------------------: |:---------------------------------------:|
|          Python           |                   3.8                   |
|          driver           |            Ascend HDK 23.0.0            |
|         firmware          |            Ascend HDK 23.0.0            |
|           CANN            |               CANN 7.0.0                |
|           torch           |               2.1.0、2.2.0               |
|         torch_npu         |             release v5.0.0              |


【Based on the current version of megatron, the performance statistics from our testing are as follows】

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Cluster Scale</th>
      <th>Precision Mode</th>
      <th>Performance </th>
      <th>Reference Performance </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="examples/aquila/README.md">Aquila</a></td>
      <td>7B</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2849 </td>
      <td> 2874 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/aquila2/README_en.md">Aquila2</a></td>
      <td>7B</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 3323 </td>
      <td> 2673 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan/README.md">Baichuan</a></td>
      <td>7B</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2685 </td>
      <td> 2036 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 1213 </td>
      <td> 862 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan2/README.md">Baichuan2</a></td>
      <td>7B</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2664 </td>
      <td> 3969 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1668 </td>
      <td> 2062 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td>7B1</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2034 </td>
      <td> 2525 </td>
    </tr>
    <tr>
      <td >176B</td>
      <td >12x8</td>
      <td> BF16 </td>
      <td> 100 </td>
      <td> 107 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/codellama/README.md">CodeLlama</a></td>
      <td>34B</td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 837 </td>
      <td> 762 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>BF16</td>
      <td> 2776 </td>
      <td> 2854 </td>
    </tr>
    <tr>
      <td >65B</td>
      <td >4x8</td>
      <td> BF16 </td>
      <td> 341 </td>
      <td> 414 </td>
    </tr>
    <tr>
      <td rowspan="5"><a href="examples/llama/README.md">LLaMA</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 3600 </td>
      <td> 3804 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 1895 </td>
      <td> 2012 </td>
    </tr>
    <tr>
        <td>33B</td>
        <td>4x8</td>
        <td>FP16</td>
        <td>621</td>
        <td>776</td>
    </tr>
    <tr>
      <td rowspan="2">65B</td>
      <td rowspan="2">4x8</td>
    </tr>
    <tr>
      <td>BF16 </td>
      <td> 348 </td>
      <td> 426 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 4200 </td>
      <td> 3850 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1990 </td>
      <td> 1920 </td>
    </tr>
    <tr>
      <td>34B</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 690 </td>
      <td> 796 </td>
    </tr>
    <tr>
      <td>70B</td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 350 </td>
      <td> 339 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/llama3/README.md">LLaMA3</a></td>
      <td>8B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2483 </td>
      <td> 2674 </td>
    </tr>
    <tr>
      <td>70B</td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 283 </td>
      <td> -- </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2499 </td>
      <td> 2867 </td>
    </tr>
    <tr>
      <td>14B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1560 </td>
      <td> 1578 </td>
    </tr>
    <tr>
      <td>72B</td>
      <td>16x8</td>
      <td>BF16 </td>
      <td> 285 </td>
      <td> 345 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/qwen15/README.md">Qwen1.5</a></td>
      <td> 7B </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  2862 </td>
      <td> 2621 </td>
      </tr>
      <tr>
      <td> 14B </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1717 </td>
      <td> 1702 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/yi/README_en.md">Yi</a></td>
      <td>34B</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 809 </td>
      <td> 730 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 487 </td>
      <td> 610 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mistral/README.md">Mistral</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2806 </td>
      <td> 2734 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/gemma/README.md">Gemma</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2938 </td>
      <td> 2607 </td>
    </tr>
  </tbody>
</table>

---


### <span id="jump18">Acceleration Features
ModelLink supports various acceleration algorithms such as tensor parallelism, pipeline parallelism, sequence parallelism, recomputation, distributed optimizer, and more. The table below shows the enable switches corresponding to each acceleration feature:

|         Acceleration Feature         |        Enable Parameter        |
|:------------------------------------:|:------------------------------:|
|           Tensor Parallel            |  --tensor-model-parallel-size  |
|          Pipeline Parallel           | --pipeline-model-parallel-size |
|       Dynamic division for PP        |        --num-layer-list        |
|          Sequence Parallel           |      --sequence-parallel       |
|            Recomputation             |    --recompute-granularity     |
|        Distributed Optimizer         |  --use-distributed-optimizer   |
|        overlap DDP allreduce         |     --overlap-grad-reduce      |
|        overlap DDP allgather         |     --overlap-param-gather     |
|           Flash attention            |        --use-flash-attn        |
|            Fused rmsnorm             |      --use-fused-rmsnorm       |
|             Fused swiglu             |       --use-fused-swiglu       |
|                 mc2                  |     --use-mc2                  |
| Fused rotary <br/>position embedding |   --use-fused-rotary-pos-emb   |
|        Sliding Window Attention      |        --sliding-window        |



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
    --overlap-param-gather \
    --use-fused-rotary-pos-emb \
    --use-mc2 \
    --sliding-window 4096 \
    ... \
    ...
```


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


## Acknowledgments

---
ModelLink is jointly contributed by the following departments of Huawei Corporation:
- Ascend Computing Product Unit
- Algorithm Unit of Computing Product Unit
- Research Unit of Computing Product Unit
- Open Computing Kit of Computing Product Unit
- General Development Department
- Global Technical Service Department

We appreciate every PR from community, and welcome to contribute to ModelLink.

## Appendix

---
- Safety Statement: [Safety Statement](https://gitee.com/ascend/ModelLink/wikis/%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)