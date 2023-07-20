# Install
```shell
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
python3 setup.py install
```

# Usage
After Installation, you can use it directly

```python
import torch
import torch_npu
from ascendspeed.mpu.ops import FlashAttention

```

forward:
```python
flash = FlashAttention()
score = flash(query, key, value, head_num, ...)
```

backward:
```python
query.requires_grad = True
score = flash(query, key, value, head_num, ...)
score.backward()
query_grad = query.grad
```
