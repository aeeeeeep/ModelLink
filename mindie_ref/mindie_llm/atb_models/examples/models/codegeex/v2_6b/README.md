# CodeGeeX2-6B 模型推理指导 <!-- omit in toc -->

# 概述

- [CodeGeeX2-6B](https://github.com/THUDM/CodeGeeX2) 是多语言代码生成模型 [CodeGeeX](https://github.com/THUDM/CodeGeeX) ([KDD’23](https://arxiv.org/abs/2303.17568)) 的第二代模型。不同于一代 CodeGeeX（完全在国产华为昇腾芯片平台训练） ，CodeGeeX2 是基于 [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) 架构加入代码预训练实现，得益于 ChatGLM2 的更优性能，CodeGeeX2 在多项指标上取得性能提升（+107% > CodeGeeX；仅60亿参数即超过150亿参数的 StarCoder-15B 近10%）。
- 此代码仓中实现了一套基于NPU硬件的CodeGeeX2推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了CodeGeeX2-6B模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE | TGI |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|--------------|----------|--------|--------|-----|
| CodeGeeX2-6B    | 支持world size 1,2  | 支持world size 1,2      | 是   | 否   | 否              | 是              | 否       | 否           | 否       | 否     | 是     | 是  |

- 此模型仓已适配的模型版本
  - [CodeGeeX2-6B](https://huggingface.co/THUDM/codegeex2-6b/tree/main)


# 使用说明

- 参考[此README文件](../../chatglm/v2_6b/README.md)

## 精度测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../../tests/modeltest/README.md)