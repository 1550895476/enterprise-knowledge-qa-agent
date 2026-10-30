from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agent import KnowledgeQAAgent
from src.runtime_config import ModelConfigError, RuntimeModelConfig


SCENARIOS = [
    {
        "name": "01-normal",
        "question": "密码忘了怎么办",
        "faults": {},
        "expected_status": "completed",
        "min_citations": 1,
    },
    {
        "name": "02-conflict",
        "question": "报销期限多久",
        "faults": {},
        "expected_status": "needs_confirmation",
        "min_citations": 2,
    },
    {
        "name": "03-out-of-scope",
        "question": "北京地铁 1 号线今天几点停运",
        "faults": {},
        "expected_status": "insufficient_evidence",
        "max_citations": 0,
        "require_gaps": True,
    },
    {
        "name": "04-paraphrase",
        "question": "费用报账最晚什么时候提交",
        "faults": {},
        "expected_status": "needs_confirmation",
        "min_citations": 2,
    },
    {
        "name": "05-context-follow-up",
        "question": "那研发员工呢？",
        "history": [
            {"role": "user", "content": "普通员工国内出差餐补是多少？"},
            {"role": "assistant", "content": "普通员工国内出差餐补为 100 元/天。"},
        ],
        "faults": {},
        "expected_status": "completed",
        "min_citations": 1,
    },
    {
        "name": "06-tool-failure",
        "question": "密码忘了怎么办",
        "faults": {"fail_search_once": True},
        "expected_status": "completed",
        "min_citations": 1,
        "require_failed_tool": True,
    },
]


def _record(scenario: dict[str, Any], config: RuntimeModelConfig) -> dict[str, Any]:
    model = config.build_model()
    agent = KnowledgeQAAgent.from_default_data(model=model, **scenario["faults"])
    history = scenario.get("history", [])
    result = agent.run(scenario["question"], history=history)
    payload = result.to_dict()
    trace = payload.pop("trace")
    public_config = config.public()
    public_config.pop("api_key_configured", None)
    return {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_config": public_config,
        "retrieval_config": {
            "dense_backend": agent.tools.knowledge_base.embedder.backend,
            "sparse_backend": "okapi-bm25",
            "evidence_policy": "read_original_before_answer",
        },
        "input": {"question": scenario["question"], "history": history, **scenario["faults"]},
        "expected_status": scenario["expected_status"],
        "actual_output": payload,
        "trace": trace,
    }


def _validate_record(scenario: dict[str, Any], record: dict[str, Any]) -> list[str]:
    output = record["actual_output"]
    problems: list[str] = []
    if output.get("status") != scenario["expected_status"]:
        problems.append(f"预期状态 {scenario['expected_status']}，实际 {output.get('status')}")
    citation_count = len(output.get("citations", []))
    if citation_count < scenario.get("min_citations", 0):
        problems.append(f"引用不足：至少 {scenario['min_citations']} 条，实际 {citation_count} 条")
    if citation_count > scenario.get("max_citations", float("inf")):
        problems.append(f"不应返回引用，实际 {citation_count} 条")
    if scenario.get("require_gaps") and not output.get("gaps"):
        problems.append("应返回资料缺口，但 gaps 为空")
    if scenario.get("require_failed_tool") and not any(event.get("status") == "failed" for event in record["trace"]):
        problems.append("应包含工具失败事件，但 Trace 中没有 status=failed")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="使用当前在线模型录制评审所需的真实 Agent Trace")
    parser.add_argument("--output-dir", default="examples/online", help="输出目录，默认 examples/online")
    args = parser.parse_args()

    config = RuntimeModelConfig()
    try:
        config.build_model()
    except ModelConfigError as exc:
        print(f"无法录制真实在线 Trace：{exc}")
        print("请先复制 .env.example 为 .env，填写在线模型配置；API Key 不会写入输出文件。")
        return 2
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    recorded_files: list[str] = []

    for scenario in SCENARIOS:
        try:
            record = _record(scenario, config)
        except Exception as exc:
            message = f"{scenario['name']}: 模型调用失败：{exc}"
            failures.append(message)
            print(message)
            print("录制已停止；请检查网络或中转站后重新运行，已有文件不会被伪造为成功结果。")
            break
        target = output_dir / f"{scenario['name']}.json"
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        recorded_files.append(target.name)
        actual = record["actual_output"]["status"]
        print(f"{target}: {actual}")
        failures.extend(f"{scenario['name']}: {problem}" for problem in _validate_record(scenario, record))

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_config": config.public(),
        "files": recorded_files,
        "supplemental_files": (
            ["07-document-replacement.json"]
            if (output_dir / "07-document-replacement.json").exists()
            else []
        ),
        "failures": failures,
    }
    manifest["model_config"].pop("api_key_configured", None)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        print("存在与预期不一致的真实模型结果，请检查 Trace 后调整模型或 Prompt：")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("在线模型 Trace 录制完成，文件中不包含 API Key。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
