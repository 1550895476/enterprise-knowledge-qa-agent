# 最终提交检查清单

本文件区分自动完成项与必须由候选人本人完成的事项。不要提交真实 API Key。

## 自动检查

```bash
python -m unittest discover -s tests -v
python -m scripts.check_submission
```

## 必须手动完成

1. 在 `.env` 中填写自己的在线模型 Base URL、API Key 和模型名；Web 页面可另行测试和启用模型，但页面配置只存在服务内存中，独立的 Trace 脚本读取 `.env`；
2. 运行 `python -m scripts.record_online_traces`，检查 `examples/online/` 中六类必验场景均有真实运行记录，且至少两条记录确实显示模型根据工具结果改变动作；
3. 如果真实模型输出状态与预期不一致，检查 Trace，调整 Prompt 或更换模型后重新录制，不能手工伪造输出；
4. 在 `docs/05-retrospective.md` 填写真实投入时间，并根据实际情况更新未完成项；
5. 临时修改或上传一份内容不同的制度，重新提问并确认答案、引用和后续动作会随原文变化；保留对应 Trace 后恢复合成测试数据；
6. 确认 `.env`、上传文件和真实企业资料没有被 Git 跟踪；
7. 将最终修改提交并推送 GitHub，记录完整 commit SHA；
8. 私有仓库必须手动邀请指定评审账号并确认其具有读取权限；
9. 按下方格式提交题号、仓库地址和最终 SHA。

## 最终提交信息模板

```text
题号：3（企业知识问答 Agent）
GitHub URL：https://github.com/1550895476/enterprise-knowledge-qa-agent
最终 commit SHA：<完成最终提交后填写 40 位 SHA>
```
