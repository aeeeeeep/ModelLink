# CHANGE LOG

## Version 1.0

- 首版本发布
- 支持模型Aquila、Baichuan、Baichuan2、Bloom、InternLM、LLaMA、LLaMA2、Qwen、Mixtral模型预训练对话和评估。

## Version 1.1

- Megatron配套升级至core r0.6.0分支，新版本--overlap-param-gather参数只支持mcore模型。
- MindSpeed配套升级至2b0edd2 commitid
- 新增支持模型LLaMA3、Qwen1.5-72B、Mistral-7B、Gemma模型
- jit-compile参数由环境变量控制转为--jit-compile脚本参数控制，不配置默认为false。
