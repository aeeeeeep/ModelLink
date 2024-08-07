## ModelLink 测试用例贡献说明

### 看护列表
<table>
    <tr>
        <th>Tests</th>
        <th>Module</th>
        <th>Submodule</th>
        <th>Scripts</th>
        <th>Accuracy</th>
        <th>Throughput</th>
        <th>Memory</th>
    </tr>
    <tr>
        <td rowspan="2">ST</td>
        <td rowspan="2">Pretrain</td>
        <td>TP，PP，VPP，重计算，enable-recompute-layers-per-pp-rank</td>
        <td>shell_scripts/llama2_tp2_pp4_vpp2.sh</td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>CP，分布式优化器，ReuseFP32Param，FusedRMSNorm，FusedSwiGlu，FusedRope，overlap-grad-reduce、overlap-param-gather</td>
        <td>shell_scripts/llama2_tp2_cp4_mem_recompute.sh</td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="8">UT</td>
        <td rowspan="3">CP</td>
        <td>hybrid</td>
        <td>dist_algo/test_hybrid_context_parallel.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ring_attn</td>
        <td>dist_algo/test_ringattn_context_parallel.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ulysses</td>
        <td>dist_algo/test_ulysses_context_parallel.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">model_module</td>
        <td>rope</td>
        <td>embeddings/test_rotary_pos_embedding.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>transformer_attention</td>
        <td>transformer/test_attention.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="3">checkpoint</td>
        <td>mcore_dynamic</td>
        <td>test_convert_ckpt_from_huggingface.py</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>mcore_vpp</td>
        <td>test_convert_ckpt_from_huggingface.py</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>legacy_dynamic</td>
        <td>test_convert_ckpt_from_huggingface.py</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Pipeline</td>
        <td colspan="6"></td>
    </tr>
</table>


其余的再补充...

### 目录说明

#### st:

(1) 看护特性：
**预训练功能**，主要包含各种**分布式并行特性算法**组合：

① TP, PP, VPP

② CP, EP（待上库）

(2) 规则说明：

##### 校验规则：

① 加载权重提取前 15 步 **lm loss**，与 `baseline_results` 文件夹下的标杆数据进行对比。loss 在非确定性计算情况下单步差异在 0.01 以内；

② 提取最后 10 步 **throughput**（消除 warm_up 影响）与 `baseline_results` 文件夹下的标杆数据进行平均吞吐对比，吞吐比标杆高可以直接通过，若劣化则保证劣化比在 %5 以内；

③ 提取 8 卡的 **allocated memory** 与 **max allocated memory** 与`baseline_results` 文件夹下的标杆进行显存占用比对。给定分布式并行切分策略，理论上每个 rank 的显存分配是固定的，显存占用比标杆低可以直接通过，若劣化则保证劣化比在 10% 以内。

##### 其余规则：

① 贡献脚本用例请放置于 `st/shell_scripts` 文件夹下，命名规则为 **{模型名}_{切分策略}**，如 `llama2_tp2_pp4_vpp2_ptd.sh`，请贡献者严格对齐；

② 标杆数据请放置于 `st/baseline_results` 文件夹下，**命名保证完全与 shell 脚本对齐**，否则自动化脚本执行将扫描不到；

③ 如需增加看护指标，在 `st/st_utils/common.py` 下补充对应正则表达式与相应提取逻辑，在 `st/st_utils/test_ci_pipeline.py` 补充对应函数逻辑；

④ 为保证门禁执行时间，无特别场景，尽量不要增加 shell 脚本用例，建议在原有用例基础上补充该部分逻辑；若需要补充将审视其必要性。

#### ut：

所有通过 `pytest` 拉起的用例建议放置于本文件夹。

`ckpt`：权重转换；

`process_data`：数据处理相关；

`dist_algo`：分布式并行算法前向逻辑，例如现有 `CP` 特性用例；

`model_module`：与模型结构计算相关逻辑

...

#### pipeline:

待补充说明...