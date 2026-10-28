from __future__ import annotations

import argparse
import json
import sys

from .agent import KnowledgeQAAgent
from .decision import OpenAICompatibleDecisionModel


def main() -> int:
    parser = argparse.ArgumentParser(description="企业知识问答 Agent")
    parser.add_argument("--question", required=True)
    parser.add_argument("--history-file", help="包含 role/content 对象数组的 JSON 文件")
    parser.add_argument("--model", choices=["offline", "openai"], default="offline")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--fail-search-once", action="store_true")
    parser.add_argument("--fail-read-once", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    history = []
    if args.history_file:
        with open(args.history_file, encoding="utf-8") as file:
            history = json.load(file)
    model = OpenAICompatibleDecisionModel() if args.model == "openai" else None
    agent = KnowledgeQAAgent.from_default_data(
        model=model,
        max_steps=args.max_steps,
        fail_search_once=args.fail_search_once,
        fail_read_once=args.fail_read_once,
    )
    try:
        result = agent.run(args.question, history=history)
    except Exception as exc:  # CLI boundary: expose actionable failure, not a traceback-only result.
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())

