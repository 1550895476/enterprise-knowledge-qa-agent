# 最终提交检查清单

本文件区分自动完成项与必须由候选人本人完成的事项。不要提交真实 API Key。

## 自动检查

```bash
python -m unittest discover -s tests -v
python -m scripts.check_submission
```

## 完成状态

- [x] 使用候选人的在线模型配置录制真实 Trace；API Key 仅保存在被 Git 忽略的 `.env` 中；
- [x] 六类必验场景均有真实模型运行记录，并体现不同工具观察后的不同动作路径；
- [x] 对不符合预期的真实模型输出通过 Prompt、证据协议和测试修正，没有手工伪造 Trace；
- [x] `docs/05-retrospective.md` 已填写实际投入时间和未完成项；
- [x] 已上传两份内容不同的制度并保留前后对比 Trace：`examples/online/07-document-replacement.json`；
- [x] `.env`、临时上传文件和真实密钥均未被 Git 跟踪；
- [x] 代码已提交并推送 GitHub；最终 SHA 应在最后一次提交后通过 `git rev-parse HEAD` 获取；
- [ ] 私有仓库必须由候选人手动邀请指定评审账号并确认其具有读取权限；
- [ ] 将题号、仓库地址和命令输出的最终 SHA 填入实际投递表单。

## 最终提交信息模板

```text
题号：3（企业知识问答 Agent）
GitHub URL：https://github.com/1550895476/enterprise-knowledge-qa-agent
最终 commit SHA：运行 `git rev-parse HEAD`，将输出的 40 位 SHA 填入投递表单（Git commit 无法在自身内容中自引用最终 SHA）。
```
