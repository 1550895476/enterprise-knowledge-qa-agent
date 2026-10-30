# BGE-M3 启用与完整验证记录

## 结论

2026-10-30 已在本项目的 `.venv` 中启用真实 BGE-M3 与 BGE-Reranker。模型探针返回 `embedding_backend=bge-m3`、1024 维归一化向量和 `reranker_backend=bge-reranker`；42 项自动化测试全部通过，六条真实在线模型 Trace 全部通过。运行记录没有使用 hashing Dense 降级，也没有写入 API Key。

## 环境

- Python 3.12.7（项目 `.venv`，不是系统 Anaconda）
- Windows 11
- PyTorch 2.15.0+cpu
- Transformers 4.57.6
- FlagEmbedding 1.3.5
- BGE-M3：1024 维 Dense embedding
- BGE-Reranker：交叉编码重排
- Sparse：Okapi BM25

安装与启用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
$env:ENABLE_BGE='1'
$env:HF_HUB_DISABLE_XET='1'
```

权重已缓存后，验证和启动使用离线模型缓存，避免 Hugging Face 网络波动；这不会关闭在线问答模型的网络：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
```

## 模型探针

实际结果：

```json
{
  "embedding_backend": "bge-m3",
  "dimension": 1024,
  "norm": 1.0,
  "reranker_backend": "bge-reranker",
  "related_score": 0.993256,
  "unrelated_score": 0.000016
}
```

相关片段分数显著高于无关片段，证明当前进程实际执行了 BGE-Reranker，而不是词项分数占位。

## 42 项自动化测试

```powershell
$env:ENABLE_BGE='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

实际结果：`Ran 42 tests`，`OK`，进程退出码 0，完整运行耗时 105.09 秒。原始控制台记录保存于 `examples/bge-m3-test-output.txt`。

测试覆盖明确答案、文档冲突、资料外问题、改写问题、上下文追问、工具失败、步数上限、上传文件问答、商品消歧、Dense 与 HyDE 独立动作、BM25、RRF、重排、引用溯源及模型接口容错。证据协议额外验证：有候选必须先读原文；无候选必须返回资料缺口，不能无依据追问。

## 六条真实在线模型 Trace

```powershell
$env:ENABLE_BGE='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -m scripts.record_online_traces
```

实际结果：6/6 通过，进程退出码 0，完整运行耗时 252.76 秒。记录位于 `examples/online/`，每条均包含：

- 真实在线模型决策来源 `openai_compatible_model`；
- `dense_backend=bge-m3`；
- `sparse_backend=okapi-bm25`；
- 模型动作、工具参数、观察摘要、下一动作与决策依据；
- 最终状态、资料缺口以及可核验的原文引用。

结果为：明确答案 `completed`；冲突 `needs_confirmation` 且引用两份原文；资料外 `insufficient_evidence` 且零引用；改写问题仍发现冲突；上下文追问 `completed`；工具首次失败后恢复并 `completed`。

补充的资料替换记录位于 `examples/online/07-document-replacement.json`：原制度回答 24 个月，修订制度回答 6 个月，引用文档 ID 随资料变化；两轮均使用 `bge-m3`、`okapi-bm25` 和 `bge-reranker`。

## 网络中断处理

首次下载时 Hugging Face Xet 连接不稳定，完整权重下载并校验后放入标准缓存。后续设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，BGE 启动只读本地缓存；在线大模型仍通过项目配置的 OpenAI-compatible 接口调用。这样能区分“检索模型本地可用”和“问答中转站网络可用”两类故障。
