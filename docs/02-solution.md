# 方案与技术取舍

## 调用流程

```text
问题 + 历史
  ↓
决策模型选择 search/read/ask/finish
  ↓
工具执行并返回结构化 observation
  ↓
更新已命中文档、已读取原文、失败次数
  ↓
决策模型根据 observation 选择下一步
  ↺ 直到回答、追问或达到上限
```

`src/agent.py` 只负责状态和循环，不负责判断答案；动作选择由 `src/decision.py` 的模型完成。这样工具结果确实会影响下一步：无命中会改写查询，命中候选会读取原文，多个候选会继续交叉核对，读取失败会改读另一份文档。

## 动态决策

离线模型不是预先写死的 Trace，而是一个可替换的决策模型接口。每次 `choose` 都接收完整状态与刚完成的工具观察，再生成结构化 Action。它用于无密钥复现和测试；`OpenAICompatibleDecisionModel` 将相同状态与观察发送给兼容 Chat Completions 的模型。

冲突判断只基于已经读取的原文：同一主题出现不同天数/金额时，返回 `needs_confirmation`，引用所有读取文档，并要求补充适用范围或优先级。没有读取到原文时不把检索摘要当作最终证据。

## 技术取舍

- 选择 Python 标准库：3 小时考核中降低安装和环境风险，保留清晰的模型/工具边界；
- 不引入向量数据库：合成资料规模小，词项 + 同义词检索足够展示检索改写与证据链；
- 不使用多 Agent：题目重点是单 Agent 的动态决策，多 Agent 会增加噪声而不增加验收覆盖；
- JSON Trace：便于评审复制运行、替换文档并对比动作变化；
- `max_steps=6`：防止真实模型或工具异常时无限循环。

## 增强链路：商品知识库

`src/advanced_agent.py` 提供第二条 LangGraph 链路：

1. `ProductMatcher` 结合历史会话识别商品别名，并与 `data/products.json` 标准商品库匹配；多个相似型号返回选择项，出现型号语义但无法确认时直接追问，不调用检索。
2. `StructureAwareChunker` 按 PDF 页、标题层级、Layout 坐标和图片引用生成父 Chunk/子 Chunk；过短文本可在切片阶段合并，图片通过 `image_refs` 保留，图片摘要器可注入真实多模态模型。
3. `HybridRetriever` 并行计算稠密、稀疏、HyDE 和 MCP 路线，使用 RRF 去重，再由 BGE-Reranker 精排，按累计概率动态截取 Top-p。
4. 每个候选同时携带版本日期、发布时间、适用范围、页码、章节、父子 Chunk ID 和图片路径。可信度用于排序但不能覆盖正文冲突；冲突检测返回事实、不同值和 Chunk ID。
5. `MilvusVectorStore`、`MongoTraceStore` 在配置连接串时接入真实服务，否则保留相同接口的内存实现。`requirements-ml.txt` 启用 BGE-M3/BGE-Reranker 权重。

这条链路复用原有 Agent 的“模型动作—工具—观察—下一动作”循环，因此搜索候选不会直接成为答案，必须读取 Chunk 原文后再生成引用。

## 上传文件并提问

`DocumentUploadService` 完成以下链路：

```text
multipart 文件 → 类型/大小/PDF 头校验 → 隔离保存
→ PDF 或文本解析 → 结构感知父子 Chunk
→ 动态加入 HybridRetriever/Milvus 适配层
→ 返回 document_id 与 ready 状态
→ /ask 携带 document_ids → 请求级受限检索 → 原文引用回答
```

索引文件和状态保存在 `runtime/uploads/<document_id>/`，启动时自动恢复。`document_id` 来自 SHA-256，可对相同内容去重。检索工具按请求创建独立作用域实例，商品范围和上传文档范围通过 LangGraph 状态显式传递，避免并发请求串库。

## Web 操作界面与模型配置

根路径 `/` 返回无需构建工具的单页界面，支持模型配置、文件上传、文档勾选、多轮问答和引用展示。`RuntimeModelConfig` 提供线程安全的运行时配置：离线模式使用 `OfflineDecisionModel`，OpenAI-compatible 模式使用用户填写的 Base URL、模型名和 API Key。读取配置时永不返回密钥原文，密钥只驻留进程内存；服务重启后需重新填写或使用环境变量。

在线模型负责检索动作与证据回答，Prompt 要求只能使用已读取的 `readings`，并引用文档 ID、页码、章节和 Chunk ID。离线模式继续使用可复现的证据生成器。
