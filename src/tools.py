from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

from .models import Document, SearchHit, ToolObservation


STOP_WORDS = {
    "什么", "怎么", "如何", "多少", "多久", "一下", "请问", "帮我", "查询", "规定",
    "员工", "需要", "可以", "吗", "呢", "的", "是", "那", "我", "想", "知道",
}

SYNONYMS = {
    "报销": {"报销", "费用报销", "报账"},
    "期限": {"期限", "时限", "多久", "多长时间", "几天", "提交时间"},
    "餐补": {"餐补", "伙食补贴", "吃饭补贴", "餐费补贴"},
    "密码": {"密码", "口令", "登录密码"},
    "忘记": {"忘记", "忘了", "找回", "重置"},
    "研发": {"研发", "技术", "开发"},
    "开机": {"开机", "启动", "开机启动"},
    "电源键": {"电源键", "开关键", "电源按钮"},
    "分辨率": {"分辨率", "清晰度"},
    "配网": {"配网", "联网", "网络配置"},
    "保修": {"保修", "保修期", "质保", "质保期"},
}


def normalize_terms(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    terms: set[str] = set()
    for token in cleaned.split():
        if token not in STOP_WORDS:
            terms.add(token)
            for canonical, variants in SYNONYMS.items():
                if token in variants:
                    terms.add(canonical)
                    terms.update(variants)
    # Chinese questions are often not space separated; preserve useful substrings.
    for canonical, variants in SYNONYMS.items():
        if any(variant in cleaned for variant in variants):
            terms.add(canonical)
            terms.update(variants)
    return terms


class KnowledgeBase:
    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.documents = [Document(**item) for item in raw]

    def search(self, query: str, limit: int = 5) -> ToolObservation:
        terms = normalize_terms(query)
        scored: list[SearchHit] = []
        for doc in self.documents:
            searchable = " ".join([doc.title, doc.applicability, doc.content, *doc.tags]).lower()
            matched = {term for term in terms if term in searchable}
            if not matched:
                continue
            score = len(matched) / max(len(terms), 1)
            snippet = doc.content[:180] + ("…" if len(doc.content) > 180 else "")
            scored.append(SearchHit(
                doc.doc_id, doc.title, doc.version_date, doc.applicability, snippet, round(score, 3),
                page_start=doc.page_start,
                page_end=doc.page_end,
                chapter=doc.chapter,
                chunk_id=doc.chunk_id or doc.doc_id,
                parent_chunk_id=doc.parent_chunk_id,
                metadata={"source_type": doc.source_type, "image_refs": doc.image_refs},
            ))
        scored.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return ToolObservation(
            tool="search_documents",
            ok=True,
            summary=f"命中 {len(scored[:limit])} 份文档",
            data={"hits": [hit.__dict__ for hit in scored[:limit]], "query_terms": sorted(terms)},
        )

    def read(self, doc_id: str) -> ToolObservation:
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return ToolObservation(
                    tool="read_document",
                    ok=True,
                    summary=f"已读取 {doc_id} 原文片段",
                    data={
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "version_date": doc.version_date,
                        "applicability": doc.applicability,
                        "content": doc.content,
                        "page_start": doc.page_start,
                        "page_end": doc.page_end,
                        "chapter": doc.chapter,
                        "chunk_id": doc.chunk_id or doc.doc_id,
                        "parent_chunk_id": doc.parent_chunk_id,
                        "image_refs": doc.image_refs,
                        "source_type": doc.source_type,
                    },
                )
        return ToolObservation("read_document", False, f"文档 {doc_id} 不存在", {"doc_id": doc_id})


class FaultInjectingTools:
    """Adds one-shot failures for reproducible evaluation without changing the KB."""

    def __init__(self, knowledge_base: KnowledgeBase, fail_search_once: bool = False, fail_read_once: bool = False):
        self.knowledge_base = knowledge_base
        self.fail_search_once = fail_search_once
        self.fail_read_once = fail_read_once

    def search(self, query: str) -> ToolObservation:
        if self.fail_search_once:
            self.fail_search_once = False
            return ToolObservation("search_documents", False, "模拟检索服务暂时不可用", {"retryable": True})
        return self.knowledge_base.search(query)

    def read(self, doc_id: str) -> ToolObservation:
        if self.fail_read_once:
            self.fail_read_once = False
            return ToolObservation("read_document", False, "模拟文档读取服务暂时不可用", {"retryable": True, "doc_id": doc_id})
        return self.knowledge_base.read(doc_id)

