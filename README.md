# 企业知识问答 Agent

一个面向企业制度与产品 FAQ 的可离线运行问答 Agent。它不会只做一次检索，而是由决策模型在每轮观察工具结果后选择：补充检索、读取原文、追问或结束。

## 1. 环境与运行

要求 Python 3.10+。项目只使用标准库，安装依赖不是必须的：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

默认使用离线决策模型：

```bash
python -m src.cli --question "报销期限多久"
python -m src.cli --question "普通员工出差餐补标准是多少？"
```

输出 JSON，包含 `answer`、`citations`、`gaps`、`status` 与 `trace`。用 `--pretty` 可获得适合人工阅读的格式。

可选地使用 OpenAI-compatible Chat Completions 服务：

```bash
copy .env.example .env
# 填写 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL
python -m src.cli --model openai --question "报销期限多久"
```

程序不会读取或提交真实密钥；没有模型服务时请使用默认的 `offline` 入口。离线入口验证流程和规则，不等同于真实大模型效果。

模拟工具故障以复现容错路径：

```bash
python -m src.cli --question "报销期限多久" --fail-search-once
python -m src.cli --question "密码忘了怎么办" --fail-read-once
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 2. 评审入口

建议按以下顺序查看：

1. `src/agent.py`：主循环与状态上限；
2. `src/decision.py`：离线模型和 OpenAI-compatible 模型的动作选择；
3. `src/tools.py`：文档检索、片段读取和故障注入；
4. `examples/`：真实运行输出与 Trace；
5. `docs/02-solution.md`、`docs/03-evaluation.md`：设计取舍与验收结果。

## 3. 项目结构

```text
src/                 Agent、模型适配器、知识库工具、CLI
tests/               可执行单元测试与验收测试
examples/            输入、实际输出、Trace
data/                合成知识库
docs/                题目分析、方案、评估、AI 使用记录、复盘
```

## 4. 设计边界

- 资料冲突时不会自行选择一个版本；会列出相互冲突的原文并请求确认适用制度。
- 找不到资料时会给出资料缺口和 `insufficient_evidence` 状态，不编造答案。
- 工具错误会进入下一轮决策；连续失败或达到步数上限会返回 `incomplete`。
- 文档是合成材料，不包含真实企业隐私。
