# 评估与验证结果

运行命令：

```bash
python -m unittest discover -s tests -v
```

当前验收测试覆盖 42 项：基础链路的正常答案、冲突、资料外、改写问题、上下文追问、两类工具失败、步数上限，高级链路的商品消歧、embedding/BM25 混合检索和结构感知切片，上传问答与上传安全校验，Web 根页面和模型配置脱敏，未配置在线模型时的拒绝行为、DeepSeek 模型连通性测试、中转站模型列表发现、Responses API 的协议、推理强度、JSON 输出参数和 Agent 动作解析、带推理文本或 Markdown 包装的 JSON 动作解析、示例引用原文完整性、真实问答进度流的阶段与顺序、HyDE 复用当前模型、无关短问题经过证据门槛后不引用无关文档、Okapi BM25 排序和基础知识库 embedding/BM25 分数，以及临时 TLS/网络错误重试与认证错误不重试。新增两项证据协议测试：有检索候选但尚未读取原文时，模型不能直接回答或追问；完整检索无命中时，模型不能返回无依据追问，必须标记资料不足。检索测试明确核对原问题 embedding Dense、原问题 BM25 Sparse、HyDE 生成、HyDE Dense 检索、RRF、重排和冲突检查为独立动作，并验证 HyDE 失败时记录失败、跳过该路召回且流程继续。

| 验收场景 | 材料 |
| --- | --- |
| 正常问答：密码重置 | `examples/01-normal-*` |
| 冲突：报销 30 天 vs 15 天 | `examples/02-conflict-*` |
| 资料外：地铁停运 | `examples/03-out-of-scope-*` |
| 改写问题：费用报账期限 | `examples/04-paraphrase-*` |
| 上下文追问：研发员工餐补 | `examples/05-context-*` |
| 工具失败：检索一次失败后恢复 | `examples/06-tool-failure-*` |
| 高级商品问答：Pro X1 开机 | `examples/07-advanced-product.json` |
| 资料替换：保修期 24 个月改为 6 个月 | `examples/online/07-document-replacement.json` |

## 预期与实际

- 正常问答：实际状态为 `completed`，引用 D04 原文；
- 冲突问答：实际状态为 `needs_confirmation`，同时引用 D01、D02，不输出单一期限；
- 资料外：实际状态为 `insufficient_evidence`，无引用并有资料缺口；
- 改写问题：实际仍命中报销文档并发现冲突；
- 上下文追问：实际读取 D05，回答研发员工 150 元/天；
- 工具失败：实际 Trace 记录失败，随后改写查询并完成回答。
- 资料替换：同一问题在原制度下回答 24 个月并引用原文档，切换到修订制度后回答 6 个月并引用新的文档 ID；两次动作均为 `search → read → finish`，说明资料改变后答案和引用改变，而证据充分性相同，因此动作序列无需人为制造差异。

高级链路额外通过 3 项测试：

- `Pro X1 怎么开机`：解析为 P100-X1，引用第 3 页、章节和 Chunk ID，答案为长按 3 秒；
- `Pro 怎么开机`：识别 P100-X1/P100-X2 两个候选，只追问，不执行知识库检索；
- `型号 Z9 怎么开机`：无法确认标准商品，直接拒答，不执行知识库检索。

高级检索测试还核对了 `dense`、`sparse`、`hyde` 三路 RRF 路线字段，Dense 与 HyDE 在进度和 Trace 中各自独立，且包含 Metadata 可信度字段、页码和父子 Chunk 字段。运行时输出不包含不存在的 MCP 路线。

上传链路测试实际上传两份保修政策，分别限定 `document_ids` 提问，验证 24 个月与 6 个月答案不会相互串库；同时覆盖不支持扩展名和超过大小上限两个失败场景。

浏览器验证覆盖根页面加载、模型切换弹窗、Provider/协议/推理强度/输出格式切换、在线模型测试与启用、文档选择、问题发送、回答状态，以及包含文档 ID、定位信息和原文片段的引用卡片。根页面返回 200，不再跳转到 Swagger 文档。

模型测试使用 Mock HTTP 响应验证 DeepSeek 的 Chat Completions 请求发送到 `/chat/completions`，并核对返回模型名、延迟和响应摘要；另一路验证 Responses API 请求发送到 `/responses`，且请求体包含所选推理强度与 JSON 输出约束。真实中转站的可用性需由用户在界面中使用自己的密钥测试。

模型发现测试模拟中转站 `/models` 返回重复模型，验证模型 ID 去重排序、完整 Chat Completions URL 自动归一化为 Base URL，以及最终请求路径为 `/v1/models`。

已使用 `python -m scripts.record_online_traces` 将当前在线模型的正常、冲突、资料外、改写问题、上下文追问和工具失败六条路径写入 `examples/online/`。六条记录均保留 `decision_source=openai_compatible_model`、`dense_backend=bge-m3` 和 `sparse_backend=okapi-bm25`；冲突与改写路径各引用两份原文，资料外路径无引用且返回资料缺口，工具失败路径记录 `status=failed` 后恢复。测试桩记录只证明流程，`examples/online/` 用于验证真实模型效果。BGE 环境、探针、耗时和完整命令见 `docs/07-bge-validation.md` 与 `examples/bge-m3-validation.json`。

另使用 `python -m scripts.record_replacement_trace` 录制资料替换前后对比。两轮均由真实在线模型决策并使用 BGE-M3/BGE-Reranker；答案从 24 个月变为 6 个月，引用源文档 ID 同步变化，验证回答不是写死在代码或预设日志中。

## 限制与失败风险

1. 自动化测试使用确定性测试桩验证 Agent 循环和安全边界，不证明真实模型的泛化能力；真实模型效果需要使用目标模型补充运行记录；
2. 本次提交验证已启用真实 BGE-M3；若显式设置 `ENABLE_BGE=0`，开发环境会使用标注为 `hashing-ngram` 的轻量 Dense 降级。Sparse 路线使用 Okapi BM25，大规模数据仍应迁移到 Milvus 的稠密/稀疏索引；
3. 冲突抽取目前针对天数/金额这类简单数值，复杂法律条款冲突仍需更强的结构化解析；
4. OpenAI-compatible 接口依赖外部服务稳定性，超时会由 CLI 返回错误，不会伪造答案；
5. 运行记录中的版本日期是合成数据日期，不代表真实制度。
6. Milvus 的 collection schema、MongoDB 索引和 BGE 模型部署参数仍需根据生产集群做迁移配置；本仓库的降级实现只用于本地验证。
7. 上传接口目前没有登录、租户隔离、病毒扫描和对象存储；本地目录持久化适合考核演示，不应直接作为生产文件存储。

审核时可将 `data/documents.json` 中 D01 或 D02 的数值替换，再运行测试/CLI，观察引用和冲突状态是否随原文变化。
