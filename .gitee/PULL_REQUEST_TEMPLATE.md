<!--  Thanks for sending a pull request!  Here are some tips for you:

If this is your first time, please read our contributor guidelines: https://gitee.com/ascend/community/blob/master/CONTRIBUTING.zh.md
-->

**What type of PR is this?**
> Uncomment only one ` /kind <>` line, hit enter to put that in a new line, and remove leading whitespaces from that line:
>
> /kind bug
> /kind task
> /kind feature

**What does this PR do / why do we need it**:
> xxx

**Which issue(s) this PR fixes**:
<!--
*Automatically closes linked issue when PR is merged.
Usage: `Fixes #<issue number>`, or `Fixes (paste link of issue)`.
-->
> Fixes #

**Special notes for your reviewers**:
> xxx

**CheckList**:
<!--
自检通过，[ ] 修改为 [x]
-->

- [ ] PR标题和描述是否按格式填写
- [ ] 若涉及外接口变更，是否已通过变更评审
- [ ] 是否通过本地IDE对代码进行静态检查
- [ ] 是否通过本地IDE对代码进行格式化处理

- [ ] 是否进行空指针校验
- [ ] 是否进行返回值校验 (禁止使用void屏蔽安全函数、自研函数返回值，C++标准库函数确认无问题可以屏蔽)
- [ ] 是否正确释放new/malloc申请的内存
- [ ] 是否充分考虑接口的异常场景
- [ ] 是否正确记录错误日志

- [ ] 新增或修改的代码是否完成DT测试，且测试覆盖率达到标准要求

**Test Report**:
<!--
Optional, customized as needed.
可选，各团队根据需要自定制格式，描述关键信息，应该可以用截图.
-->