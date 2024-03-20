# 模型说明
1. 将transformer 框架下deepseek-moe-16b模型迁移到Ascend
2. 使能TP+EP并行推理方式
3. 使能多机推理功能，支持多机TP、TP+PP等

# Deepseek推理流程
 
## 环境安装
 
1. 下载代码到本地服务器
 
2.  配置环境
CANN、驱动等安装流程直接参考官网指导：安装说明-昇腾软件安装指南-Atlas 800T A2 训练服务器-...-文档首页-昇腾社区 (hiascend.com)
~~~
##  python3.8
conda create -n test python=3.8
conda activate test

##  安装 torch 和 torch_npu
pip3 install torch==2.1.0   #  arm
#  pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu  # (x86)
pip3 install torch-npu==2.1.0

#  使能 ascend-toolkit 
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 

#  安装transformers库
pip install transformers==4.38.2

#  安装CANN、分布式依赖
pip3 install attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py
 ~~~
3. 下载deepseek-MOE-base权重
hugging face直接下载：deepseek-ai/deepseek-moe-16b-base at main (huggingface.co) , 存放到本地服务器路径（示例）: /data/deepseek_16B/
 
## 脚本启动
~~~
1. 修改启动脚本 run.sh
weight_dir="/data/deepseek_16B/"
2. 修改启动文件run_deepseek.py
os.environ["MASTER_ADDR"] = "xxxx"    # 主节点服务器IP，单机情况下设置为localhost
3. 启动推理测试（默认单机8卡+TP并行）
cd deepseek-MOE
bash run.sh
~~~
 
## 其他修改说明
~~~
1. 使能EP
修改模型文件modeling_deepseek_pipe_parallel.py, 设置
self.expert_parallel = True    #  line 361
2. 修改并行大小
修改模型文件modeling_deepseek_pipe_parallel.py，设置 
initialize_model_parallel(8, 1, 1)  #  tensor_model_parallel_size，pipeline_model_parallel_size，sequence_parallel_size,    at line 1058
3. 多机PP推理
参照修改2，设置模型并行初始化参数。之后，修改启动脚本run_deepseek.py，以双机并行为例：
#  修改world_size为实际卡数
WORLD_SIZE=16
#  节点2设置rank为8-15
for((RANK_ID=$RANK_ID_START;RANK_ID<$((8+RANK_ID_START));RANK_ID++));
do
export LOCAL_RANK=$RANK_ID
export RANK=`expr $RANK_ID + 8`
export WORLD_SIZE=$WORLD_SIZE
bind=${map["$RANK_ID"]}
echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
numactl --cpunodebind=$bind --membind $bind python3 run_deepseek.py --load_path $weight_dir &
done
wait
4. 依据下面链接设置绑核参数
https://www.hiascend.com/document/detail/zh/canncommercial/700/foundmodeldev/foundmodelinfer/atlaslmimog_0058.html
 ~~~

