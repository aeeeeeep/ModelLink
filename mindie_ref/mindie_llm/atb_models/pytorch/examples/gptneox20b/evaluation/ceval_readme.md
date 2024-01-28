# 精度测试指南

## 1. 下载测试数据集

https://huggingface.co/datasets/ceval/ceval-exam/tree/main

## 2. 下载测试脚本

下载数据脚本：
https://github.com/hkust-nlp/ceval/blob/main/subject_mapping.json

## 3. 导入模型配置文件

1、新建工作目录work_dir，并在目录下创建测试结果目录test_result
2、将`pytorch/examples/atb_speed_sdk/atb_speed`整个目录全部拷贝到work_dir
3、将`pytorch/examples/gptneox20b/test/`下的所有文件拷贝到work_dir
4、将`pytorch/examples/gptneox20b/patches/models/`下的模型文件modeling_gpt_neox_model.py文件拷贝至work_dir
5、将下载的测试数据集进行解压后的数据和脚本上传至work_dir
6、修改配置文件文件model_path参数改成为模型权重存放目录
示例（work_dir）：  
--atb_speed
--ceval-exam (包含：数据文件夹dev、test、val三者)
--test_result (测试结果目录)
--config.ini
--configuration_gpt_neox.py
--modeling_gpt_neox_model.py (模型脚本)
--sdk_ceval_test_npu.py (性能测试脚本)
--subject_mapping.json (数据脚本)  
--tmp_summary.py

### 4. 运行脚本

使用 `python sdk_ceval_test_npu.py` 进行精度测试(如果使用GPU则运行python sdk_ceval_test_gpu.py)

结束后在test_result目录下查看测试结果。

| 文件                        | 用途                   | 
|---------------------------|----------------------| 
| device0.log               | 运行过程日志               |
| cache0.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| result_0_classes_acc.json | 测试数据下按不同维度统计准确率      |
| result_0_subject_acc.json | 测试数据下按不同学科统计准确率      |

### 4. 精度计算

将work_dir下的tmp_summary.py文件CSV_PATH参数改为结果目录test_result的路径，python tmp_summary.py 运行后得到精度绝对值。


