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

强化学习等特性持续研发中....

### 支持模型
当前ModelLink支持下列模型的预训练以及微调:

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>序列</th>
      <th>预训练</th>
      <th>推理</th>
      <th>LoRA</th>
      <th>SFT</th>
      <th>对话</th>
      <th>评估</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="examples/aquila/README.md">Aquila</a> </td>
      <td> 7B </td>
      <td> 2K </td>
      <td> <a href="examples/aquila/pretrain_aquila_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/aquila/generate_aquila_7b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/aquila/evaluate_aquila_7b_ptd.sh">eval</a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/aquila2/README.md">Aquila2</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td> <a href="examples/aquila2/pretrain_aquila2_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/aquila2/generate_aquila2_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/aquila2/evaluate_aquila2_7b_ptd.sh">eval</a> </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td> 4K </td>
      <td> <a href="examples/aquila2/pretrain_aquila2_34b_ptd_16p.sh"> pretrain </a> </td>
      <td> <a href="examples/aquila2/generate_aquila2_34b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/aquila2/evaluate_aquila2_34b_ptd.sh">eval</a> </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="examples/baichuan/README.md">Baichuan</a> </td>
      <td> 7B </td>
      <td> 4K </td>
      <td> <a href="examples/baichuan/pretrain_baichuan_ptd_7B.sh"> pretrain </a> </td>
      <td> <a href="examples/baichuan/generate_baichuan_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/baichuan/evaluate_baichuan_7B_ptd.sh">eval</a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> 4K </td>
      <td> <a href="examples/baichuan/pretrain_baichuan_ptd_13B.sh"> pretrain </a> </td>
      <td> <a href="examples/baichuan/generate_baichuan_13b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/baichuan/evaluate_baichuan_13B_ptd.sh"> eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="examples/baichuan2/README.md">Baichuan2</a> </td>
      <td> 7B </td>
      <td> 4K </td>
      <td> <a href="examples/baichuan2/pretrain_baichuan2_ptd_7B.sh"> pretrain </a> </td>
      <td> <a href="examples/baichuan2/generate_baichuan2_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/baichuan2/evaluate_baichuan2_7B_ptd.sh">eval</a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> 4K </td>
      <td> <a href="examples/baichuan2/pretrain_baichuan2_ptd_13B.sh"> pretrain </a> </td>
      <td> <a href="examples/baichuan2/generate_baichuan2_13b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/baichuan2/evaluate_baichuan2_13B_ptd.sh">eval</a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td> 7B1 </td>
      <td> 2K </td>
      <td> <a href="examples/bloom/pretrain_bloom_ptd_7B.sh"> pretrain </a> </td>
      <td> <a href="examples/bloom/generate_bloom_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/bloom/evaluate_bloom_7B_ptd.sh">eval</a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td> 176B </td>
      <td> 2K </td>
      <td> <a href="examples/bloom/pretrain_bloom_176b.sh"> pretrain </a> </td>
      <td> <a href="examples/bloom/generate_bloom_176b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/bloom/evaluate_bloom_176b_ptd.sh">eval</a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="1"> <a href="examples/chatglm3/README.md">ChatGLM3</a> </td>
      <td> 6B </td>
      <td> 8K </td>
      <td> <a href="examples/chatglm3/pretrain_chatglm3_6B_8K.sh"> pretrain </a> </td>
      <td> <a href="examples/chatglm3/generate_chatglm3_6B.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/chatglm3/evaluate_chatglm3_6B.sh">eval</a> </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="1"> <a href="examples/codellama/README.md">CodeLlama</a> </td>
      <td> 34B </td>
      <td> 4K </td>
      <td> <a href="examples/codellama/pretrain_codellama_34b_ptd_16p.sh"> pretrain </a> </td>
      <td> <a href="examples/codellama/generate_codellama_34b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/codellama/evaluate_codellama_34b_ptd.sh">eval</a> </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td> <a href="examples/intern/pretrain_internlm_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/intern/generate_internlm_7b_ptd.sh">generate</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/intern/evaluate_internlm_7B_ptd.sh">eval</a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td>65B</td>
      <td> 2K </td>
      <td> <a href="examples/intern/pretrain_internlm_65b_ptd.sh"> pretrain </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama/README.md">LLaMA</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td> <a href="examples/llama/pretrain_llama_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama/generate_llama_7b_ptd.sh"> generate </a> </td>
      <td> <a href="examples/llama/tune_llama_7b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama/evaluate_llama_7B_ptd.sh"> eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 2K </td>
      <td> <a href="examples/llama/pretrain_llama_13b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama/generate_llama_13b_ptd.sh">generate </a> </td>
      <td> <a href="examples/llama/tune_llama_13b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama/evaluate_llama_13B_ptd.sh">eval</a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td>33B</td>
      <td> 2K </td>
      <td> <a href="examples/llama/pretrain_llama_33B_ptd_32p.sh"> pretrain </a> </td>
      <td> <a href="examples/llama/generate_llama_33b_ptd.sh">generate </a> </td>
      <td> <a href="examples/llama/tune_llama_33b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama/evaluate_llama_33B_ptd.sh">eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td > 65B </td>
      <td> 2K </td>
      <td> <a href="examples/llama/pretrain_llama_65b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama/generate_llama_65b_ptd.sh"> generate </a> </td>
      <td > <a href="examples/llama/tune_llama_65b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama/evaluate_llama_65B_ptd.sh">eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</a></td>
      <td>7B</td>
      <td> 4K </td>
      <td> <a href="examples/llama2/pretrain_llama2_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama2/generate_llama2_7b_ptd.sh">generate </a> </td>
      <td> <a href="examples/llama2/tune_llama2_7b_ptd.sh">lora </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama2/evaluate_llama2_7B_ptd.sh">eval </a>  </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 4K </td>
      <td> <a href="examples/llama2/pretrain_llama2_13B_ptd_8p.sh"> pretrain </a> </td>
      <td> <a href="examples/llama2/generate_llama2_13b_ptd.sh">generate </a> </td>
      <td> <a href="examples/llama2/tune_llama2_13b_ptd.sh">lora </a> </td>      
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama2/evaluate_llama2_13B_ptd.sh">eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td> 34B </td>
      <td> 4K </td>
      <td> <a href="examples/llama2/pretrain_llama2_34B_ptd_16p.sh"> pretrain </a> </td>
      <td> <a href="examples/llama2/generate_llama2_34B_ptd.sh"> generate </a> </td>
      <td> <a href="examples/llama2/tune_llama2_34b_ptd.sh">lora </a> </td>
      <td > -- </td>
      <td > -- </td>
      <td> <a href="examples/llama2/evaluate_llama2_34B_ptd.sh">eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td > 70B </td>
      <td> 4K </td>
      <td> <a href="examples/llama2/pretrain_llama2_70b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama2/generate_llama2_70b_ptd.sh"> generate </a> </td>
      <td > <a href="examples/llama2/tune_llama2_70b_ptd.sh">lora</a> </td>
      <td > -- </td>
      <td > -- </td>
      <td> <a href="examples/llama2/evaluate_llama2_70B_ptd.sh">eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/llama3/README.md">LLaMA3</a></td>
      <td>8B</td>
      <td> 8K </td>
      <td> <a href="examples/llama3/pretrain_llama3_8b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama3/generate_llama3_8b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama3/generate_llama3_8b_chat_ptd.sh"> chat </a> </td>
      <td> <a href="examples/llama3/evaluate_llama3_8b_ptd.sh"> eval </a>  </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td>70B</td>
      <td> 8K </td>
      <td> <a href="examples/llama3/pretrain_llama3_70b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/llama3/generate_llama3_70b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/llama3/evaluate_llama3_70b_ptd.sh"> eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen/pretrain_qwen_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen/generate_qwen_7b_ptd.sh"> generate </a></td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen/evaluate_qwen_7b_ptd.sh">eval </a></td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td>14B</td>
      <td> 2K </td>
      <td> <a href="examples/qwen/pretrain_qwen_14b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen/generate_qwen_14b_ptd.sh">generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen/evaluate_qwen_14b_ptd.sh"> eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td>72B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen/pretrain_qwen_72b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen/generate_qwen_72b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen/evaluate_qwen_72b_ptd.sh"> eval </a> </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="7"><a href="examples/qwen15/README.md">Qwen1.5</a></td>
      <td>0.5B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen15/pretrain_qwen15_0point5b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_0point5b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_0point5b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【社区】 </td>
    <tr>
      <td>1.8B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen15/pretrain_qwen15_1point8b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_1point8b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_1point8b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【社区】 </td>
    <tr>
      <td>4B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen15/pretrain_qwen15_4b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_4b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_4b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【社区】 </td>
    <tr>
      <td>7B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen15/pretrain_qwen15_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_7b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_7b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【社区】 </td>
    <tr>
      <td>14B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen15/pretrain_qwen15_14b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_14b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_14b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【社区】 </td>
    <tr>
      <td>32B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen15/pretrain_qwen15_32b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_32b_ptd.sh"> generate </a> </td>
      <td> <a href="examples/qwen15/tune_qwen15_32b_ptd.sh"> lora </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_32b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【社区】 </td>
    <tr>
      <td>72B</td>
      <td> 8K </td>
      <td> <a href="examples/qwen15/pretrain_qwen15_72b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/qwen15/generate_qwen15_72b_ptd.sh"> generate </a> </td>
      <td> <a href="examples/qwen15/tune_qwen15_72b_ptd.sh"> lora </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/qwen15/evaluate_qwen15_72b_ptd.sh"> eval </a> </td>
      <td rowspan="1"> 【昇腾】 </td>
    </tr>
    <tr>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/yi/README.md">Yi</a></td>
      <td>34B</td>
      <td> 4K </td>
      <td> <a href="examples/yi/pretrain_yi_34b_ptd_16p.sh"> pretrain </a> </td>
      <td> <a href="examples/yi/generate_yi_34b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/yi/evaluate_yi_34b_ptd.sh"> eval </a> </td>
      <td> 【社区】 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td> 32K </td>
      <td> <a href="examples/mixtral/pretrain_mixtral_8x7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/mixtral/generate_mixtral_8x7b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/mixtral/evaluate_mixtral_8x7b_ptd.sh"> eval </a>  </td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mistral/README.md">Mistral</a></td>
      <td>7B</td>
      <td> 32K </td>
      <td> <a href="examples/mistral/pretrain_mistral_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/mistral/generate_mistral_7b_ptd.sh"> generate </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/mistral/evaluate_mistral_7b_ptd.sh"> eval </a> </td>
      <td>【昇腾】</td>
    <tr>
      <td rowspan="2"><a href="examples/gemma/README.md">Gemma</a></td>
      <td>2B</td>
      <td> 8K </td>
      <td> <a href="examples/gemma/pretrain_gemma_2b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/gemma/generate_gemma_2b_ptd.sh"> generate </a></td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/gemma/evaluate_gemma_2b_ptd.sh">eval </a></td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td>7B</td>
      <td> 8K </td>
      <td> <a href="examples/gemma/pretrain_gemma_7b_ptd.sh"> pretrain </a> </td>
      <td> <a href="examples/gemma/generate_gemma_7b_ptd.sh"> generate </a></td>
      <td > <a href="examples/gemma/tune_gemma_7b_ptd.sh">lora</a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="examples/gemma/evaluate_gemma_7b_ptd.sh">eval </a></td>
      <td> 【昇腾】 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/gpt3/README.md">GPT3</a></td>
      <td>175B</td>
      <td> 2K </td>
      <td> <a href="examples/gpt3/pretrain_gpt3_175B_ptd.sh"> pretrain </a> </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> 【社区】 </td>
    </tr>
  </tbody>
</table>


### 脚本命名规则

|        脚本         |   规则   |
|:-----------------:|:------:|
|  pretrain_xxx.sh  | 预训练脚本  |
|    tune_xxx.sh    | LoRA脚本 |
|  generate_xxx.sh  |  推理脚本  |
|  xxxx_chat_xx.sh  |  对话脚本  |
| evaluation_xxx.sh |  评估脚本  |


---

## 模型版本与性能说明

上述列表中支持的模型，我们在[examples](./examples/)文件夹中提供了各模型的训练脚本和readme说明，里面有详细的模型训练、推理、评估流程。

【需要注意模型使用时的配套环境版本，参考如下】

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          driver           |        Ascend HDK 24.1.RC2          |
|         firmware          |        Ascend HDK 24.1.RC2          |
|           CANN            |            CANN 8.0.RC2             |
|           torch           |            2.1.0、2.2.0             |
|         torch_npu         |        release v6.0.RC2           |


【基于现版本我们实测的性能情况统计如下（硬件信息：Atlas 900 A2 PODc）】

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>序列</th>
      <th>集群规模</th>
      <th>精度模式</th>
      <th>性能 </th>
      <th>参考性能 </th>
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
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/aquila2/README.md">Aquila2</a></td>
      <td>7B</td>
      <td>2K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 3323 </td>
      <td> 2673 </td>
    </tr>
    <tr>
      <td>34B</td>
      <td>4K</td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 854 </td>
      <td> 732 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan/README.md">Baichuan</a></td>
      <td>7B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2685 </td>
      <td> 2036 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 1213 </td>
      <td> 862 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan2/README.md">Baichuan2</a></td>
      <td>7B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2664 </td>
      <td> 3969 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1668 </td>
      <td> 2062 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td>7B1</td>
      <td>2K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2034 </td>
      <td> 2525 </td>
    </tr>
    <tr>
      <td>176B</td>
      <td>2K</td>
      <td >12x8</td>
      <td> BF16 </td>
      <td> 100 </td>
      <td> 107 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/chatglm3/README.md">ChatGLM3</a></td>
      <td>6B</td>
      <td> 8K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 4297 </td>
      <td> 4267 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/codellama/README.md">CodeLlama</a></td>
      <td>34B</td>
      <td>4K</td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 837 </td>
      <td> 762 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>BF16</td>
      <td> 2776 </td>
      <td> 2854 </td>
    </tr>
    <tr>
      <td >65B</td>
      <td>2K</td>
      <td >4x8</td>
      <td> BF16 </td>
      <td> 341 </td>
      <td> 414 </td>
    </tr>
    <tr>
      <td rowspan="5"><a href="examples/llama/README.md">LLaMA</td>
      <td>7B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 3600 </td>
      <td> 3804 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 1895 </td>
      <td> 2012 </td>
    </tr>
    <tr>
        <td>33B</td>
        <td>2K</td>
        <td>4x8</td>
        <td>FP16</td>
        <td>621</td>
        <td>776</td>
    </tr>
    <tr>
      <td rowspan="2">65B</td>
      <td rowspan="2">2K</td>
      <td rowspan="2">4x8</td>
    </tr>
    <tr>
      <td>BF16 </td>
      <td> 348 </td>
      <td> 426 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</td>
      <td>7B</td>
      <td>4K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 4200 </td>
      <td> 3850 </td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1990 </td>
      <td> 1920 </td>
    </tr>
    <tr>
      <td>34B</td>
      <td>4K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 749 </td>
      <td> 796 </td>
    </tr>
    <tr>
      <td>70B</td>
      <td>4K</td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 420 </td>
      <td> 430 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/llama3/README.md">LLaMA3</td>
      <td>8B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2483 </td>
      <td> 2674 </td>
    </tr>
    <tr>
      <td>70B</td>
      <td> 8K </td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 283 </td>
      <td> 355 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2499 </td>
      <td> 2867 </td>
    </tr>
    <tr>
      <td>14B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1560 </td>
      <td> 1578 </td>
    </tr>
    <tr>
      <td>72B</td>
      <td> 8K </td>
      <td>16x8</td>
      <td>BF16 </td>
      <td> 285 </td>
      <td> 345 </td>
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
      <tr>
      <td> 1.8B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 13029 </td>
      <td> 12181 </td>
      <tr>
      <td> 4B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 5033 </td>
      <td> 5328 </td>
      <tr>
      <td> 7B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2862 </td>
      <td> 2621 </td>
      <tr>
      <td> 14B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1717 </td>
      <td> 1702 </td>
      <tr>
      <td> 32B </td>
      <td> 8K </td>
      <td> 4x8 </td>
      <td> BF16 </td>
      <td> 751 </td>
      <td> 708 </td>
      <tr>
      <td> 72B </td>
      <td> 8K </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> 301 </td>
      <td> 317 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/yi/README.md">Yi</a></td>
      <td>34B</td>
      <td> 4K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 809 </td>
      <td> 730 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td> 32K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 487 </td>
      <td> 610 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mistral/README.md">Mistral</a></td>
      <td>7B</td>
      <td> 32K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2806 </td>
      <td> 2734 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/gemma/README.md">Gemma</a></td>
      <td>2B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 6821 </td>
      <td> 7602 </td>
    </tr>
    <tr>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2938 </td>
      <td> 2607 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/gpt3/README.md">GPT3</a></td>
      <td>175B</td>
      <td> 2K </td>
      <td> 16x8 </td>
      <td> FP16 </td>
      <td> 153 </td>
      <td> -- </td>
    </tr>
  </tbody>
</table>

---

## <span id="jump18"> 加速算法与融合算子
ModelLink支持张量并行、流水线并行、序列并行、重计算、分布式优化器等多种加速算法和融合算子，下表为各种加速特性对应的使能开关：

|                 加速特性                 |              使能参数              |
|:------------------------------------:|:------------------------------:|
|                 张量并行                 |  --tensor-model-parallel-size  |
|                流水线并行                 | --pipeline-model-parallel-size |
|               流水并行动态划分               |        --num-layer-list        |
|                 序列并行                 |      --sequence-parallel       |
|                 重计算                  |    --recompute-granularity     |
|                分布式优化器                |  --use-distributed-optimizer   |
|           DDP allreduce 掩盖           |     --overlap-grad-reduce      |
|           Flash attention            |        --use-flash-attn        |
|            Fused rmsnorm             |      --use-fused-rmsnorm       |
|             Fused swiglu             |       --use-fused-swiglu       |
|                 mc2                  |           --use-mc2            |
| Fused rotary <br/>position embedding |   --use-fused-rotary-pos-emb   |
|       Sliding window attention       |        --sliding-window        |


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
