from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from src.advanced_agent import EnterpriseKnowledgeAgent
from src.document_service import DocumentUploadService
from src.runtime_config import ModelConfigError, RuntimeModelConfig


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "examples" / "online" / "07-document-replacement.json"
MANIFEST = ROOT / "examples" / "online" / "manifest.json"
QUESTION = "这份制度规定的保修期是多久？"


def _actions(trace: list[dict[str, Any]]) -> list[str]:
    return [
        str(event["action"])
        for event in trace
        if event.get("kind") in {"tool_call", "decision"} and event.get("action")
    ]


def _source_ids(output: dict[str, Any]) -> list[str]:
    return [
        str(item.get("source_doc_id") or item.get("doc_id"))
        for item in output.get("citations", [])
    ]


def _run(agent: EnterpriseKnowledgeAgent, model: Any, document_id: str) -> dict[str, Any]:
    result = agent.run(QUESTION, document_ids=[document_id], decision_model=model)
    payload = result.to_dict()
    trace = payload.pop("trace")
    return {"actual_output": payload, "trace": trace}


def _validate(baseline: dict[str, Any], replacement: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    baseline_output = baseline["actual_output"]
    replacement_output = replacement["actual_output"]
    if baseline_output.get("status") != "completed" or replacement_output.get("status") != "completed":
        problems.append("替换前后都必须完成回答")
    if "24个月" not in baseline_output.get("answer", ""):
        problems.append("替换前回答未体现 24 个月")
    if "6个月" not in replacement_output.get("answer", ""):
        problems.append("替换后回答未体现 6 个月")
    if not baseline_output.get("citations") or not replacement_output.get("citations"):
        problems.append("替换前后都必须包含原文引用")
    if _source_ids(baseline_output) == _source_ids(replacement_output):
        problems.append("替换前后引用文档 ID 未变化")
    for label, run in (("替换前", baseline), ("替换后", replacement)):
        if not any(event.get("decision_source") == "openai_compatible_model" for event in run["trace"]):
            problems.append(f"{label} Trace 缺少真实在线模型决策")
    return problems


def _update_manifest() -> None:
    if not MANIFEST.exists():
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    supplemental = set(manifest.get("supplemental_files", []))
    supplemental.add(TARGET.name)
    manifest["supplemental_files"] = sorted(supplemental)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    config = RuntimeModelConfig()
    try:
        model = config.build_model()
    except ModelConfigError as exc:
        print(f"无法录制替换资料 Trace：{exc}")
        return 2

    with tempfile.TemporaryDirectory(prefix="knowledge-replacement-") as runtime:
        agent = EnterpriseKnowledgeAgent()
        service = DocumentUploadService(agent.retriever, runtime_root=runtime)
        baseline_doc = service.ingest_bytes(
            "产品保修制度-2026版.md",
            "# 产品保修制度\n\n本设备保修期为24个月，自激活之日起计算。".encode("utf-8"),
            "text/markdown",
        )
        replacement_doc = service.ingest_bytes(
            "产品保修制度-修订版.md",
            "# 产品保修制度（修订版）\n\n本设备保修期调整为6个月，自激活之日起计算。".encode("utf-8"),
            "text/markdown",
        )
        baseline = _run(agent, model, baseline_doc.document_id)
        replacement = _run(agent, model, replacement_doc.document_id)

        baseline_actions = _actions(baseline["trace"])
        replacement_actions = _actions(replacement["trace"])
        problems = _validate(baseline, replacement)
        public_config = config.public()
        public_config.pop("api_key_configured", None)
        payload = {
            "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_config": public_config,
            "retrieval_config": {
                "dense_backend": agent.retriever.embedder.backend,
                "sparse_backend": "okapi-bm25",
                "reranker_backend": agent.retriever.reranker.backend,
                "evidence_policy": "read_original_before_answer",
            },
            "input": {
                "question": QUESTION,
                "baseline_document_id": baseline_doc.document_id,
                "replacement_document_id": replacement_doc.document_id,
            },
            "expected_status": "completed",
            "actual_output": replacement["actual_output"],
            "trace": replacement["trace"],
            "replacement_validation": {
                "baseline": baseline,
                "replacement": replacement,
                "baseline_actions": baseline_actions,
                "replacement_actions": replacement_actions,
                "answer_changed": baseline["actual_output"]["answer"] != replacement["actual_output"]["answer"],
                "citation_changed": _source_ids(baseline["actual_output"]) != _source_ids(replacement["actual_output"]),
                "action_sequence_changed": baseline_actions != replacement_actions,
                "action_review": (
                    "动作序列保持一致是预期结果：替换前后都只有一份充分且无冲突的原文，因此模型均执行检索、读取、结束；"
                    "答案值与引用文档随资料改变。"
                ),
                "failures": problems,
            },
        }
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _update_manifest()

    if problems:
        print("替换资料 Trace 验证失败：")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"{TARGET.relative_to(ROOT)}: 资料替换前后验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
