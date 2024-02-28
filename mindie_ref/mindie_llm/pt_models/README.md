# 介绍

pt_models旨在在昇腾CANN框架下执行pytorch大模型脚本提供样例参考，方便开发者对自己的大模型进行NPU的迁移

# 公告

- 2024年3月1号：提供llama2-70b前端切分分布式执行样例

# 环境依赖

|   软件    |             [版本](https://www.hiascend.com/zh/)             |
| :-------: | :----------------------------------------------------------: |
|  Python   |                            3.8.18                            |
|  driver   | [2023Q4商发](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/258924109?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
| firmware  | [2023Q4商发](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/258924109?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
|   CANN    | [2023Q4商发](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/258923273?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
|  kernel   | [2023Q4商发](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/258923273?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
|   torch   |                            2.1.0                             |
| torch_npu |   [2023Q4商发](https://gitee.com/ascend/pytorch/releases)    |
|   apex    | [2023Q4商发](https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.1.0/20231225.2/pytorch_v2.1.0_py38.tar.gz) |

**安装第三方依赖**

```shell
pip3 install -r requirement.txt
```

# 环境搭建

```shell
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu
pip3 install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
pip3 install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# 安装 CANN包

pip3 install -r requirements.txt 


```



# 模型及数据集





# 快速体验

```shell
cann_path=/usr/local/Ascend #昇腾cann包安装目录
model_path=xxx/llama2-70b #下载的对应模型的weight和tokenizer
bash llm_inference.sh --cann_path=${cann_path} --batch_size=1 --model_path=${model_path} --input_padding=false --device_list="0,1,2,3,4,5,6,7"
```

## 执行示例

```

```



# 执行参数介绍

| 参数             | 参数类型 | 参数说明                                                     |
| ---------------- | -------- | ------------------------------------------------------------ |
| model_path       | String   | 模型weight和tokenizer路径。必选                              |
| batch_size       | Int      | batch的大小，默认1                                           |
| seq_len_in       | Int      | 输入句子的最大长度，默认1024                                 |
| seq_len_out      | Int      | 输出句子的最大长度，默认1024                                 |
| dtype            | String   | 权重数据类型，默认fp16，支持fp16,fp32,bf16                   |
| device_list      | String   | 指定执行的device列表，默认0                                  |
| input_padding    | Bool     | 是否对输入padding到最大长度                                  |
| exe_mode         | String   | pytorch脚本执行方式，默认图模式。支持eager和dynamo           |
| jit_compile      | Bool     | 是否进行算子编译，默认false，表示使能二进制，不做算子编译    |
| cann_path        | String   | CANN包安装路径，默认/usr/local/Ascend                        |
| distributed_mode | String   | 部署方式，默认deepspeed，分布式多卡执行。                    |
| log_level        | Int      | CANN日志级别，默认为3，支持0，1，2，3表示DEBUG,INFO,WARNING,ERROR |
|                  |          |                                                              |

# 评价指标



# NPU迁移点

## 图模式迁移

```python
# transformers/generation/utils.py中greedy_search函数
exe_mode = os.getenv("EXE_MODE", "dynamo")
backend = os.getenv("BACKEND", "npu")
input_padding = bool(os.getenv("INPUT_PADDING", "True"))

if exe_mode == "dynamo":
  logging.info("Start to run model in dynamo mode, dynamic=%s, fullgraph=%s, backend=%s" % (not input_padding,
    True, backend))
  import torchair as tng
  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.experimental_config.frozen_parameter = True
  npu_backend = tng.get_npu_backend(compiler_config=config)
  self = torch.compile(self, dynamic=not input_padding, fullgraph=True, backend=npu_backend)
else:
  logging.info("Start to run model in eager(HOST API) mode")
```



## 性能优化

### 固定kv cache大小

优化原因：transformers的llama源码中对于kv cache的处理是作为模型的输入，模型中通过cat进行拼接后，返回新的kv cache，这种更新方式存在多次申请内存及拷贝的性能损失。

优化方式：根据句子最大长度申请号一块固定大小的kv cache tensor，然后通过scatter_update_算子对指定位置上的kv cache进行更新

```python
# transformers/models/llama/modeling_llama.py
# LlamaForCausalLM的prepare_inputs_for_generation函数新增逻辑
# 固定kv cache的大小，用作全量图和增量图的kv cache更新
batch_size, seq_length = input_ids.shape
use_dtype = self.model.torch_dtype

if past_key_values is None:
    past_key_values = ()
    for i in range(self.model.num_hidden_layers):
        kv_shape = (batch_size, self.model.num_key_value_heads // self.world_size, self.model.max_position_embeddings, 
            self.model.hidden_size // self.model.num_attention_heads)
        k_cache = torch.zeros(kv_shape, dtype=use_dtype, device=input_ids.device)
        v_cache = torch.zeros(kv_shape, dtype=use_dtype, device=input_ids.device)
        past_key_values += ((k_cache, v_cache),)
```

更新kv的改动

```python
# LlamaAttention的forward函数
'''
if q_len == 1:
    kv_seq_len = past_key_value[0].shape[-2]
else:
    kv_seq_len = key_states.shape[-2]
'''

cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

'''
if q_len > 1:
    tmp_ids = torch.zeros(bsz, dtype=torch.int64, device=position_ids.device)
    torch_npu.scatter_update_(past_key_value[0], tmp_ids, key_states, -2)
    torch_npu.scatter_update_(past_key_value[1], tmp_ids, value_states, -2)
else:
    position_ids = position_ids.reshape(-1) + 1
    torch_npu.scatter_update_(past_key_value[0], position_ids, key_states, -2)
    torch_npu.scatter_update_(past_key_value[1], position_ids, value_states, -2)

key_states1 = past_key_value[0] if q_len == 1 else key_states
value_states1 = past_key_value[1] if q_len == 1 else value_states

past_key_value = past_key_value if use_cache else None
'''
```

固定kv后，由于shape变化带来的其他tensor的改动

```python
# _expand_mask函数新增
if past_key_values_length > 0:
    mask = torch.cat([mask, torch.zeros(bsz, past_key_values_length - src_len, dtype=dtype, device=mask.device)], dim=-1)
    src_len = past_key_values_length
```

