[toc]
# atb-models
## 支持模型列表（按字母顺序排序）
<p>为方便更多开发者体验和使用昇腾芯片澎湃推理算力，该目录下提供了经典和主流大模型算法基于昇腾Transformer加速库的昇腾服务器推理的端到端流程，更多模型持续更新中。</p>
<strong>因使用版本差异，模型性能可能存在波动，性能仅供参考</strong></p>
<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>模式</th>
      <th>昇腾 </th>
    </tr>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="examples/Alpaca/README.md">Alpaca</a></td>
      <td> 7B </td>
      <td> FP16 </td>
      <td> 310P </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/Aquila/README.md">Aquila</a></td>
      <td> 7B </td>
      <td> FP16 </td>
      <td> 310P </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/Baichuan2/README.md">Baichuan2</a></td>
      <td> 7B </td>
      <td> FP16、Int8 </td>
      <td> 310P </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> FP16、Int8 </td>
      <td> 310P </td>
    </tr>
  </tbody>
</table>
## 开发指南
### 如何贡献
在开始贡献之前，请先阅读 [notice](CONTRIBUTING.md)，谢谢！

### 加速库安装

获取`Ascend-mindie-atb_*_cxx11abi*_linux-{arch}.run`
```
chmox +x Ascend-mindie-atb_*_cxx11abi*_linux-{arch}.run
./Ascend-mindie-atb_*_cxx11abi*_linux-{arch}.run --install --install-path=YOUR_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source YOUR_PATH/atb/set_env.sh
```

### 模型仓编译

#### 代码仓下载

```
git clone https://gitee.com/ascend/ModelLink.git
```

#### 代码编译

```
cd ascend-inference
bash scripts/build.sh
cd output/atb/
source set_env.sh
```

### 环境变量参考

#### 日志打印

加速库日志

```
ATB_LOG_TO_FILE=1
ATB_LOG_TO_STDOUT=1
ATB_LOG_LEVEL=INFO
TASK_QUEUE_ENABLE=1
ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
```

算子库日志

```
ASDOPS_LOG_TO_FILE=1
ASDOPS_LOG_TO_STDOUT=1
ASDOPS_LOG_LEVEL=INFO
```

性能提升（beta）
```
ATB_USE_TILING_CPY_STREAM=1
TASK_QUEUE_ENABLE=1
ATB_OPERATION_EXECUTE_ASYNC=1
ATB_OPSRUNNER_KERNEL_CACHE_GLOBAL_COUNT=40
```
