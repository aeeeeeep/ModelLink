# Llama2

本模块主要是llama2模型在npu上的适配迁移点介绍

# 性能优化

## 固定kv cache大小

**优化原因**：transformers的llama源码中对于kv cache的处理是作为模型的输入，模型中通过cat进行拼接后，返回新的kv cache，这种更新方式存在多次申请内存及拷贝的性能损失。

**优化方式**：根据句子最大长度申请号一块固定大小的kv cache tensor，然后通过scatter_update_算子对指定位置上的kv cache进行更新

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
    kv_seq_len = self.max_position_embeddings
else:
    kv_seq_len = key_states.shape[-2]
'''

cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

'''
if q_len > 1:
    tmp_ids = torch.zeros(bsz, dtype=torch.int64, device=position_ids.device)
else:
    tmp_ids = position_ids.reshape(-1) + 1
torch_npu.scatter_update_(past_key_value[0], tmp_ids, key_states, -2)
torch_npu.scatter_update_(past_key_value[1], tmp_ids, value_states, -2)

key_states1 = past_key_value[0] if q_len == 1 else key_states
value_states1 = past_key_value[1] if q_len == 1 else value_states

past_key_value = past_key_value if use_cache else None
'''
```

固定kv后，由于shape变化带来的其他tensor的改动

```python
# _expand_mask函数新增,主要原因是全量和增量流程对于attention_mask的shape要求不一样
if past_key_values_length > 0:
    mask = torch.cat([mask, torch.zeros(bsz, past_key_values_length - src_len, dtype=dtype, device=mask.device)], dim=-1)
    src_len = past_key_values_length
```

