# ModelTest README

ModelTest为大模型的性能和精度提供测试功能。

目前支持：PA场景，float16
功能：
1. 性能测试：指定batch，指定输入输出长度的e2e性能、吞吐，首Token以及非首Token性能。吞吐。
2. 精度测试：CEval, MMLU, BoolQ下游数据集
PA模型：
1. Llama (Llama-65B, Llama2-7B, Llama2-13B, Llama2-70B)
2. Starcoder-15.5B

# 使用说明

### 环境变量
```shell
# source cann环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source 加速库环境变量
source /usr/local/Ascend/atb/set_env.sh
# source 模型仓tar包解压出来后的环境变量
source set_env.sh
# 设置ATB_TESTDATA环境变量
export ATB_TESTDATA="[path]" # 用于存放测试结果的路径
```

### 安装python依赖
```
pip install loguru
pip install tabulate
pip install accelerate
pip install thefuzz
```

### 运行指令
```
bash run.sh pa_fp16 [performance|full_CEval|full_MMLU|full_BoolQ] ([case_pair]) [batch_size] [model_name] [weight_dir] [chip_num]

说明:
1. case_pair只在performance场景下接受输入，接收一组或多组输入，格式为[[seq_in_1,seq_out_1],...,[seq_in_n,seq_out_n]], 如[[256,256],[512,512]]
2. model_name:
    Llama-65B, Llama2-7B, Llama2-13B, Llama2-70B: llama
    Starcoder-15.5B: starcoder
3. weight_dir: 权重路径
4. chip_num: 使用的卡数
5. 运行完成后，会在控制台末尾呈现保存数据的文件夹

举例：
1. 测试Llama-70B在8卡[512, 512]场景下，16 batch的性能
bash run.sh pa_fp16 performance [[512,512]] 16 llama /path 8
1. 测试Starcoder-15.5B在8卡1 batch下游数据集BoolQ
bash run.sh pa_fp16 full_BoolQ 1 starcoder /path 8
``` 
 

## 300I DUO 运行操作说明

- 对于startcoder设置环境变量，修改/core/starcoder.py中prepare_environ函数。
```shell
os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
os.environ['LCCL_ENABLE_FALLBACK'] = "0"
```

