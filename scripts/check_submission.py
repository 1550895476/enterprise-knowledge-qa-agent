from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    ".env.example",
    "requirements.txt",
    "src",
    "tests",
    "examples",
    "data",
    "docs/01-requirements.md",
    "docs/02-solution.md",
    "docs/03-evaluation.md",
    "docs/04-ai-usage.md",
    "docs/05-retrospective.md",
]
REQUIRED_ONLINE_TRACES = {
    "01-normal.json",
    "02-conflict.json",
    "03-out-of-scope.json",
    "04-paraphrase.json",
    "05-context-follow-up.json",
    "06-tool-failure.json",
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        failures.append("缺少规定文件：" + "、".join(missing))

    online_dir = ROOT / "examples" / "online"
    online_records = []
    bge_records = []
    online_files: set[str] = set()
    if online_dir.exists():
        for path in sorted(online_dir.glob("*.json")):
            if path.name == "manifest.json":
                continue
            online_files.add(path.name)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                sources = {event.get("decision_source") for event in payload.get("trace", [])}
                if "openai_compatible_model" in sources:
                    online_records.append(path.name)
                retrieval_config = payload.get("retrieval_config", {})
                if retrieval_config.get("dense_backend") == "bge-m3":
                    bge_records.append(path.name)
                else:
                    failures.append(f"真实在线 Trace 未使用 BGE-M3：{path.name}")
                if retrieval_config.get("sparse_backend") != "okapi-bm25":
                    failures.append(f"真实在线 Trace 未使用 Okapi BM25：{path.name}")
                actual = payload.get("actual_output", {}).get("status")
                expected = payload.get("expected_status")
                if not actual or actual != expected:
                    failures.append(f"真实在线 Trace 状态不符合预期：{path.name}（预期 {expected}，实际 {actual}）")
                citations = payload.get("actual_output", {}).get("citations", [])
                if path.name in {"01-normal.json", "05-context-follow-up.json", "06-tool-failure.json"} and not citations:
                    failures.append(f"真实在线 Trace 缺少引用：{path.name}")
                if path.name in {"02-conflict.json", "04-paraphrase.json"} and len(citations) < 2:
                    failures.append(f"冲突 Trace 未同时引用至少两份原文：{path.name}")
                if path.name == "03-out-of-scope.json":
                    if citations or not payload.get("actual_output", {}).get("gaps"):
                        failures.append("资料外 Trace 必须无引用且包含资料缺口：03-out-of-scope.json")
                if path.name == "06-tool-failure.json" and not any(
                    event.get("status") == "failed" for event in payload.get("trace", [])
                ):
                    failures.append("工具失败 Trace 没有 status=failed 事件：06-tool-failure.json")
            except (json.JSONDecodeError, OSError):
                warnings.append(f"无法解析在线 Trace：{path.relative_to(ROOT)}")
    missing_online = sorted(REQUIRED_ONLINE_TRACES - online_files)
    if missing_online:
        failures.append("缺少六类必验在线 Trace：" + "、".join(missing_online))
    if len(online_records) < 2:
        failures.append("少于两条 Trace 含真实在线模型决策；运行 python -m scripts.record_online_traces")
    manifest_path = online_dir / "manifest.json"
    if online_dir.exists() and not manifest_path.exists():
        failures.append("缺少在线 Trace 清单：examples/online/manifest.json")
    elif manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("failures"):
                failures.append("在线 Trace 录制存在失败，请查看 examples/online/manifest.json 后重新录制")
            if set(manifest.get("files", [])) != REQUIRED_ONLINE_TRACES:
                failures.append("在线 Trace 清单未包含全部六类必验场景")
        except (json.JSONDecodeError, OSError):
            failures.append("无法解析在线 Trace 清单：examples/online/manifest.json")

    retrospective = (ROOT / "docs/05-retrospective.md").read_text(encoding="utf-8")
    if "【请填写实际用时】" in retrospective:
        failures.append("尚未填写实际投入时间：docs/05-retrospective.md")

    tracked = set(_git("ls-files").splitlines())
    if ".env" in tracked:
        failures.append(".env 被 Git 跟踪，可能泄露密钥")
    status = _git("status", "--porcelain")
    if status:
        warnings.append("Git 工作区仍有未提交修改")

    print("提交检查结果")
    print(f"- 必需文件：{'通过' if not missing else '失败'}")
    print(f"- 真实在线 Trace：{len(online_records)} 条")
    print(f"- BGE-M3 在线 Trace：{len(bge_records)} 条")
    print(f"- .env 未跟踪：{'通过' if '.env' not in tracked else '失败'}")
    for item in failures:
        print(f"[失败] {item}")
    for item in warnings:
        print(f"[提醒] {item}")
    if failures:
        return 1
    print(f"- 当前 commit：{_git('rev-parse', 'HEAD')}")
    print("自动检查通过；请继续确认私有仓库评审权限和最终提交信息。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
