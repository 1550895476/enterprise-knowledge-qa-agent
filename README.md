# 企业知识问答 Agent

一个面向企业制度与产品 FAQ 的可离线运行问答 Agent。它不会只做一次检索，而是由决策模型在每轮观察工具结果后选择：补充检索、读取原文、追问或结束。

在原始考核链路上增加了产品知识库增强链路：LangGraph 编排商品消歧与问答，PDF 结构感知切片保留页码、标题、Layout、图片与父子 Chunk，Milvus/MongoDB/BGE-M3/BGE-Reranker/FastAPI/Docker 均有可替换适配；没有重型依赖或外部服务时自动使用本地降级实现。

## 1. 环境与运行

要求 Python 3.10+。基础离线 CLI 可仅使用标准库；API、LangGraph、数据库适配器和 PDF 解析依赖通过 `requirements.txt` 安装：

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

# 高级链路：商品消歧 + 混合检索 + 证据引用
python -m src.cli --pipeline advanced --question "Pro X1 怎么开机"
python -m src.cli --pipeline advanced --question "Pro 怎么开机"
python -m src.cli --pipeline advanced --question "型号 Z9 怎么开机"
```

输出 JSON，包含 `answer`、`citations`、`gaps`、`status` 与 `trace`。用 `--pretty` 可获得适合人工阅读的格式。

可选地使用 OpenAI-compatible Chat Completions 或 Responses API 服务：

```bash
copy .env.example .env
# 填写 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL；协议等选项见 .env.example
python -m src.cli --model openai --question "报销期限多久"
```

程序不会读取或提交真实密钥；没有模型服务时请使用默认的 `offline` 入口。离线入口验证流程和规则，不等同于真实大模型效果。

## 1.1 API 与 Docker

安装 `requirements.txt` 后启动 API：

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"Pro X1 怎么开机\"}"
```

浏览器打开 `http://localhost:8000/` 可进入完整操作界面，包括：

- 离线、OpenAI、DeepSeek 与 OpenAI-compatible 中转站配置；
- 支持 Responses API / Chat Completions、推理强度与 JSON 输出切换；
- 可通过 `/models` 获取可用模型列表，选择并测试成功后再启用；
- PDF、TXT、Markdown 拖拽上传；
- 上传文档选择与检索范围控制；
- 多轮问答、状态提示及页码/章节/Chunk 引用展示。

在线模型的 API Key 只保存在当前服务进程内存，不写入仓库或运行时文件。开发接口文档保留在 `/docs`。

模型连接测试会根据所选协议调用 `POST <base_url>/responses` 或 `POST <base_url>/chat/completions`，可用于 OpenAI、DeepSeek、OpenAI-compatible 中转站及其他兼容厂商。DeepSeek 预设使用 `https://api.deepseek.com`；Base URL、模型名、协议、推理强度和输出格式都可修改。

推荐操作顺序：点击“配置 / 切换模型” → 选择 Provider 与协议 → 填写 Base URL 和 API Key → “读取可用模型” → 选择模型 → “测试所选模型” → 成功后“启用该模型”。若中转站未实现标准 `GET <base_url>/models`，仍可手动填写模型名称。

上传文件并针对该文件提问：

```bash
# 支持 PDF、TXT、Markdown，默认最大 20 MiB
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@./policy.pdf"

# 上传返回 document_id 后，限定只检索该文档
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"保修期是多久？","document_ids":["UPL-返回的文档ID"]}'

curl http://localhost:8000/documents/UPL-返回的文档ID/status
```

上传内容保存在 `runtime/uploads/`，该目录已加入 `.gitignore`；服务重启时会重新加载已完成索引的 Chunk。重复上传相同文件会复用同一个 `document_id`。

启动 API、Milvus 和 MongoDB：

```bash
docker compose up --build
```

`requirements-ml.txt` 用于安装 BGE-M3/BGE-Reranker 权重依赖；设置 `ENABLE_BGE=1` 后启用，未设置时使用可复现的词项/哈希向量降级实现。

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
src/advanced_agent.py 商品消歧、LangGraph 编排和增强链路
src/ingestion.py     PDF 页面解析、结构感知父子 Chunk
src/retrieval.py     BGE/Milvus 适配、混合检索、HyDE、MCP、RRF、重排、Top-p
src/storage.py       Milvus/MongoDB 适配与本地降级
src/api.py           FastAPI 服务入口
src/document_service.py 上传校验、持久化、索引和状态管理
src/runtime_config.py 运行时模型配置与密钥脱敏
src/static/index.html 可直接使用的上传与问答 Web UI
tests/               可执行单元测试与验收测试
examples/            输入、实际输出、Trace
data/                合成制度、商品库和 PDF Chunk
docs/                题目分析、方案、评估、AI 使用记录、复盘
```

## 4. 设计边界

- 资料冲突时不会自行选择一个版本；会列出相互冲突的原文并请求确认适用制度。
- 找不到资料时会给出资料缺口和 `insufficient_evidence` 状态，不编造答案。
- 工具错误会进入下一轮决策；连续失败或达到步数上限会返回 `incomplete`。
- 文档是合成材料，不包含真实企业隐私。
- 高级链路在商品型号未确认时不会执行知识库检索；多个相似型号时先返回选择项。
- 上传问答通过 `document_ids` 强制限定检索范围，避免不同用户选择的文档相互串库；当前示例未实现身份认证和租户权限，生产部署前必须补充。
