# telechat模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [基础环境搭建](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

  星辰语义大模型TeleChat是由中国电信人工智能科技有限公司研发训练的大语言模型，采用1.5万亿 Tokens中英文高质量语料进行训练。
     
- 参考实现：
  ```填写github链接
  
  ```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  
  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.0  | [固件与驱动](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/258924109?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
  | CANN（toolkit+kernels）                                     | 7.0.0   | [CANN](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/258923273?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
  | FrameworkPTAdapter (pytorch2.0.1)                                        | 5.0.0   | [PTA](https://gitee.com/ascend/pytorch/releases/tag/v5.0.0-pytorch2.0.1) | 
  | Python                                                     | 3.9.2   | -                                                            |            


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 基础环境搭建<a name="section4622531142816"></a>

1. 下载镜像。
   ```bash
   wget #填镜像下载链接
   docker load < telechat_infer.tar
   ```

2. 下载配套表中驱动固件
   ```
   #安装驱动
   chmod +x ./Ascend-hdk-310p-npu-driver_23.0.0_linux-aarch64.run
   ./Ascend-hdk-310p-npu-driver_23.0.0_linux-aarch64.run --full
   #安装固件
   chmod +x ./Ascend-hdk-310p-npu-firmware_7.1.0.3.220.run
   ./Ascend-hdk-310p-npu-firmware_7.1.0.3.220.run --full
   ```
   
3. 获取权重

   在当前文件夹下载权重，执行以下命令

   ```bash
   # 获取开源权重
   wget https://huggingface.co/Tele-AI/Telechat-7B/
   # 获取优化后的浮点权重
   wget 
   # 获取量化权重
   wget 
   ```

4. 启动容器

   修改telechat_docker_start.sh脚本中第16行冒号前路径为实际代码所在文件夹路径，冒号后修改为与冒号前一致
   ```bash
   bash telechat_docker_start.sh
   export LD_LIBRARY_PATH=/usr/local/python3.9.2/lib:${LD_LIBRARY_PATH}
   ```

5. (**可选，容器中已安装**) 下载相关python依赖

   ```
   pip install -r requirements.txt
   #下载配套表中FrameworkPTAdapter
   pip install torch_npu-2.0.1.post1-cp39-cp39-linux_aarch64.whl
   ```
   
6. (**可选，容器中已安装**) 下载安装cann-toolkit/cann-kernels

   ```
   ln -s /usr/local/python3.9.2/bin/python3 /usr/bin/python3.9
   #安装toolkit
   chmod +x ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run
   ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --full
   #安装kernels
   chmod +x ./Ascend-cann-kernels-310p_7.0.0_linux.run
   /Ascend-cann-kernels-310p_7.0.0_linux.run --install
   ```
   
7. 激活环境变量
   ```
   export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/atb/set_env.sh
   git clone https://gitee.com/ascend/Model_Link
   bash ./Model_Link/speed_infer/scripts/build.sh
   source ./Model_Link/speed_infer/output/atb_speed/set_env.sh
   export TASK_QUEUE_ENABLE=1
   export ATB_OPERATION_EXECUTE_ASYNC=1
   export ATB_USE_TILING_COPY_STREAM=1
   export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
   ```

## 模型推理<a name="section741711594517"></a>

1. 开始精度验证。

   - 场景一：使用提供的fp16权重，自行量化

       该模型在昇腾适配使用的技术栈为anti_outlier + ptq
       
       ```
       patch -p0 /usr/local/Ascend/ascend-toolkit/latest/tools/modelslim/pytorch/llm_ptq/anti_outlier/dag_utils/model_structure_process.py < model_structure_process.patch
       cp modeling_telechat_torch_npu.py $TRANSFORMER_PACKAGE_PATH/models/telechat/modeling_telechat.py
       python3 quant_calib_anti_outlier.py --level="L5" --jsonl_path="xxx" --checkpoint_path="xxx"
       ```
       - 命令参数说明：       
         -   `--level`：量化回退等级，默认为L5
         -   `--jsonl_path`：量化校准集路径
         -   `--checkpoint_path`：下载的浮点权重路径
     
   - 场景二：使用提供的int8量化权重
   
       新建test.jsonl文件
       按以下格式写入您想要提问的问题
       ```
       #example
       {"input": "编写一首与春天有关的诗歌。"}
       {"input": "根据给定的故事情节编写结局。以下是一篇故事的示例，故事情节应该具有足够的信息以使参与者编写自己的结局。“小明和小红是俩好朋友，他们一起长大，彼此之间的友谊非常稳固。他们的家庭都住在同一个小区，而且也是邻居。自从小明的父亲因为工作原因要外派到国外，小明便非常的寂寞。夜里，小明会偷偷地默默地流泪，每当他想起小红时，他就更加难过。小红察觉到小明的变化之后，决定在小明的父亲离开之前做点什么来让他感到开心一些。"}
       {"input": "以下是一段两人的对话内容：小明：老师李，我这次数学考试考了多少分？老师李：小明，你考了60分。小明：60分？我怎么可能考这么低分啊！老师李：你平时上课不认真，不做作业，只顾着玩游戏，这是应得的结果。小明：可是我可以补考吗？老师李：可以，但是你必须得用这段时间好好复习，不能再浪费时间了。根据这段对话内容回答问题：小明为什么会考这么低的分数？"}
       ```
       执行推理脚本
       ```
       cd ModelLink/speed_infer/pytorch/examples/telechat
       TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
       cp modeling_telechat_anti.py $TRANSFORMER_PACKAGE_PATH/models/telechat/modeling_telechat.py
       python3 run_precision.py --jsonl_path=xx --model_path=xx --quant_path=xx
       ```
       - 命令参数说明：       
         -   `--jsonl_path`：按规定格式编辑完成的jsonl文件路径
         -   `--model_path`：下载的优化后的浮点权重
         -   `--quant_path`：下载的量化权重
         
   **效果方面，完成了和基于A10卡模型量化效果对齐**
 
2. 开始性能验证

   ```
   python3 run_perf.py --model_path=xx --quant_path=xx
   ```
   
      - 命令参数说明：       
        -   `--model_path`：下载的优化后的浮点权重
        -   `--quant_path`：下载的量化权重
 
   脚本中一共测试了25组case，其中部分case具体比对效果如下：
   
    | 输入输出信息                           | NPU (tokens/s)    |   GPU (tokens/s)   |    
    | ------------------------------------ | ------- | -------|
    | 输入100输出100                         | 15  | 21 |
    | 输入1000输出100                        | 13   | 24 |
    | 输入2000输出100                        | 11   | 19 |    
    | 25组case平均                           |  13  |  18 | 
