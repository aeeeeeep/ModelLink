## ModelLink 测试用例贡献说明

### 看护列表
<table>
    <tr>
        <th>Tests</th>
        <th>Module</th>
        <th>Structure</th>
        <th>Features</th>
        <th>Scripts</th>
        <th>Accuracy</th>
        <th>Throughput</th>
        <th>Memory</th>
    </tr>
    <tr>
        <td rowspan="7">ST</td>
        <td rowspan="5">Pretrain</td>
        <td>Mcore</td>
        <td>TP，PP，VPP，重计算，enable_recompute_layers_per_pp_rank</td>
        <td><a href="st/shell_scripts/llama2_tp2_pp4_vpp2_ptd.sh">llama2_tp2_pp4_vpp2.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>CP，分布式优化器，reuse_fp32_param，recompute_activation_function, fused_rmsnorm，fused_swiglu，fused_rope，overlap_grad_reduce, overlap_param_gather</td>
        <td><a href="st/shell_scripts/llama2_tp2_cp4_mem_recompute.sh">llama2_tp2_cp4_mem_recompute.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>EP，CP，num_experts，moe_router_topk，aux_loss，moe_allgather，group_query_attention，rotary_base</td>
        <td><a href="st/shell_scripts/mixtral_mcore_tp4_cp2_ep2_ptd.sh">mixtral_mcore_tp4_cp2_ep2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Legacy</td>
        <td>TP，PP，VPP，SP，全重计算，fused_rmsnorm，fused_swiglu，fused_rope，overlap_grad_reduce</td>
        <td><a href="st/shell_scripts/llama2_tp2_pp4_vpp2_legacy.sh">llama2_tp2_pp4_vpp2_legacy.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Legacy</td>
        <td>TP，PP，EP，MoeAllGather，FusedRMSNorm，Topk，AuxLoss</td>
        <td><a href="st/shell_scripts/mixtral_legacy_tp2_ep4_ptd.sh">mixtral_legacy_tp2_ep4_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">LoRA</td>
        <td>Legacy</td>
        <td>CCLoRA, lora_fusion, 全重计算</td>
        <td><a href="st/shell_scripts/tune_llama2_tp8_pp1_lora_ptd.sh">tune_llama2_tp8_pp1_lora_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">FullSFT</td>
        <td>Legacy</td>
        <td>prompt_type</td>
        <td><a href="st/shell_scripts/tune_qwen7b_tp8_pp1_full_ptd.sh">tune_qwen7b_tp8_pp1_full_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="9">UT</td>
        <td>Inference</td>
        <td>Legacy</td>
        <td>greedy_search,  deterministic_computation</td>
        <td><a href="ut/inference/tune_inference.py">test_inference.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Evaluation</td>
        <td>Legacy</td>
        <td>mmlu</td>
        <td><a href="ut/inference/tune_evaluate.py">test_evaluate.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>CP</td>
        <td>Mcore</td>
        <td>hybrid, ring_attn, ulysses</td>
        <td><a href="ut/dist_algo/context_parallel">context_parallel</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">ModelModule</td>
        <td>Mcore</td>
        <td>rope</td>
        <td><a href="ut/model_module/embeddings/test_rotary_pos_embedding.py">test_rotary_pos_embedding.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Mcore, Legacy</td>
        <td>transformer_attention</td>
        <td><a href="ut/model_module/transformer/test_attention.py">test_attention.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Checkpoint</td>
        <td>Mcore, Legacy</td>
        <td>mcore_dynamic, mcore_vpp, legacy_dynamic</td>
        <td><a href="ut/checkpoint">checkpoint</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
	<tr>
        <td rowspan="3">ProcessData</td>
        <td rowspan="3">Mcore, Legacy</td>
        <td>pretrain_data_handler, pretrain_merge_datasets</td>
        <td><a href="ut/process_data">process_data</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
	<tr>
        <td>instruction_data_handler, instruction_merge_datasets</td>
        <td><a href="ut/process_data/test_process_instruction_data.py">test_process_instruction_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
	<tr>
        <td>instruction_data_alpaca,
        instruction_data_alpaca_history,
        instruction_data_sharegpt,
        instruction_data_openai,</td>
        <td><a href="ut/process_data/test_process_instruction_data_lf.py">test_process_instruction_data_lf.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Pipeline</td>
        <td colspan="7"></td>
    </tr>
</table>



### 开发规则

#### ST

① 贡献脚本用例请放置于 `st/shell_scripts` 文件夹下，命名规则为 **{模型名}_{切分策略}** 或者 **{模型名}_{特性名称}**， 如 `llama2_tp2_pp4_vpp2_ptd.sh`，请贡献者严格对齐；

② 注意脚本用例中不需要单独重定向log，日志收集工作已在 `run.sh` 中统一管理；

③ 标杆数据请放置于 `st/baseline_results` 文件夹下，**命名保证完全与 shell 脚本对齐**，否则自动化脚本执行将扫描不到；

④ 获取标杆数据：通过门禁任务执行获得首次数据，并将结果保存至本地 log 或者 txt 文件中，后通过本地执行 `st/st_utils/common.py` 中的 `transfer_logs_as_json` 函数进行提取，最后再连同用例脚本上仓即可；

⑤ 在贡献时候需要考虑最终校验的具体指标，精度、性能、显存，在对应指标空白处填上 `Y`，如无校验的保留空白即可。


#### UT

① 建议所有 UT 用例通过分布式 `pytest` 来拉起，即继承 tests/common.py 文件下的 `DistributedTest`，指定 `world_size`，具体参照已有用例即可；

② 建议按照功能特性进行文件夹命名区分，至多不超过两层目录，所有用例以 `test` 作为命名前缀；

③ 新增用例可以在原有用例基础上做 `test_xxx` 的补充，尽量保证测试功能的集成性；对于存在 .json 文件的用例，贡献时在 .json 中加入 `test_xxx` 配置，然后在 .py 中通过 `@pytest.mark.parameterize` 传入参数、构造用例，**请注意 .json 中的 key 值命名需与 .py 中的 test_xxx 保持统一**；

④ 在贡献时候需要考虑最终校验的具体指标，精度、性能、显存，在对应指标空白处填上 `Y`，如无校验的保留空白即可。



#### Pipeline

待补充说明...
