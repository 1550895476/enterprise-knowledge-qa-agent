from __future__ import annotations

import json
from typing import Any


GROUNDED_SYSTEM_PROMPT = """你是企业知识库问答生成器。只能使用提供的可信证据回答，不得补充证据外事实。
回答必须带文档 ID、页码、章节和 Chunk ID；如果证据不足就拒答；如果证据冲突，必须说明冲突并请求确认，不能自行选择一个版本。"""


class GroundedAnswerGenerator:
    """Evidence-window builder for grounded answer generation."""

    def build_prompt(self, question: str, evidence: list[dict[str, Any]], conflicts: list[dict[str, Any]] | None = None) -> str:
        context = [
            {
                "doc_id": item.get("doc_id"),
                "page": f"{item.get('page_start')}-{item.get('page_end')}",
                "chapter": item.get("chapter"),
                "chunk_id": item.get("chunk_id"),
                "version_date": item.get("version_date"),
                "content": item.get("content"),
            }
            for item in evidence
        ]
        return f"{GROUNDED_SYSTEM_PROMPT}\n问题：{question}\n证据：{json.dumps(context, ensure_ascii=False)}\n冲突：{json.dumps(conflicts or [], ensure_ascii=False)}"

    def generate(self, question: str, evidence: list[dict[str, Any]], conflicts: list[dict[str, Any]] | None = None) -> str:
        if not evidence:
            return "证据不足，无法可靠回答。"
        if conflicts:
            return "证据存在冲突：" + "；".join(
                f"{item['fact']} 的候选值为 {', '.join(item['values'])}（Chunk: {', '.join(item['chunk_ids'])}）"
                for item in conflicts
            ) + "。请确认适用型号或生效版本后再确定答案。"
        primary = evidence[0]
        return (
            f"根据 {primary['doc_id']} 第 {primary.get('page_start')} 页《{primary.get('chapter')}》"
            f"（Chunk {primary.get('chunk_id')}，版本 {primary.get('version_date')}），{primary['content']}"
        )

