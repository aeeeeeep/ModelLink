**介绍**

Ascend-MindIE-MindIE_llm-ATB_Models，欢迎各位开发者


**贡献要求**

开发者提交的模型包括源码、readme，并遵循以下标准


**一、源码**

1 使用者和贡献者可以直接克隆仓库，相关指令参考：

    ```
    git clone https://gitee.com/ascend/ModelLink.git
    ```

2 参考[sample-c++](https://gitee.com/ascend/ModelLink/tree/master/mindie_ref/mindie_llm/atb_models/models/chatglm2/6b) [sample](https://gitee.com/ascend/ModelLink/tree/master/mindie_ref/mindie_llm/atb_models/pytorch/examples/chatglm2_6b)

3 贡献者模型代码目录规则："ModelLink/mindie_ref/mindie_llm/atb_models/models/"（模型加速库代码）、"ModelLink/mindie_ref/mindie_llm/atb_models/pytorch/examples/"（模型modeling代码、readme、脚本）

4 从其他开源迁移的代码，请增加License声明


**二、Readme**

readme用于指导用户理解和部署样例，要包含如下内容：

1 简介：模型的来源及原理；
2 推理环境准备：昇腾插件与驱动的依赖；
3 快速上手：环境变量设置，依赖的第三方软件包和库，以及安装方法；模型复现的步骤，在线推理入口请封装成`.py`；


**三、编程规范**

规范标准
1、C++代码遵循google编程规范：Google C++ Coding Guidelines；单元测测试遵循规范： Googletest Primer。

2、Python代码遵循PEP8规范：Python PEP 8 Coding Style；单元测试遵循规范： pytest


**四、提交**

1 提交：请贡献者fork到个人仓库，通过Pull Request合入主仓的master分支。pr请勾选 “接受Pull Request时使用扁平化(Squash)合并”。

2 请在提交代码之前签署CLA协议，“个人签署”，[签署链接](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930)
    如您完成签署，可在自己提交的PR评论区输入`/check-cla`进行核实校验

3 PR合入需要通过仓库流水门禁，请在自己提交的PR评论区输入`compile-infer`拉起。

