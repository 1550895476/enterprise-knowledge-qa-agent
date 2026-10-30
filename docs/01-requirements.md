# 需求与成功标准

## 用户与目标

用户是需要查制度或产品 FAQ 的企业员工。目标是在资料充分时给出可核验答案，在资料冲突、不足或工具异常时明确暴露不确定性。

## 输入与材料

- 输入：问题，可选对话历史；
- 材料：`data/documents.json` 中的合成 FAQ/制度，每份包含文档 ID、标题、版本日期、适用范围和原文；
- 工具：`search_documents` 检索候选文档，`read_document` 读取原文。

## 输出

每次运行返回：

- `answer`：明确回答、追问或不可回答说明；
- `citations`：实际读取过的文档 ID、标题、版本日期、适用范围和原文；
- `gaps`：资料缺口或工具故障；
- `status`：`completed`、`needs_confirmation`、`insufficient_evidence` 或 `incomplete`；
- `trace`：动作、工具参数、返回摘要、后续动作和决策依据。

## 成功标准

1. 至少经历一次“模型选择动作 → 工具执行 → 观察 → 模型选择下一动作”；
2. 明确答案必须能被引用的原文支持；
3. 冲突资料必须列出冲突并请求确认，不擅自择一；
4. 资料外问题必须返回缺口而非编造；
5. 能处理改写问题、上下文追问和工具失败；
6. 有执行步数上限，超限返回明确未完成状态。

## 技术栈补充

高级实现对应 Python、LangGraph、Milvus、MongoDB、BGE-M3、BGE-Reranker、Docker、FastAPI。问答决策必须配置真实在线模型；Milvus、MongoDB 与 BGE 可通过环境变量启用，未配置时仅这些检索/存储组件使用本地降级实现。

## 用户上传文档

- 支持 PDF、TXT、Markdown，默认限制 20 MiB；
- 上传后生成内容哈希型 `document_id`，保存原文件、索引状态和 Chunk；
- `/ask` 接收 `document_ids`，只允许从指定且状态为 `ready` 的上传文档检索；
- 回答引用包含源文档 ID、页码、章节、Chunk ID 和原文；
- 文件类型不支持、文件过大、解析失败或文档未就绪时返回明确错误。
