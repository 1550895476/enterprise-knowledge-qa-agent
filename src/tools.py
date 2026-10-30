from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import Document, SearchHit, ToolObservation
from .search_backends import BM25Index, BGEEmbeddingProvider, cosine_similarity, normalize_terms


class KnowledgeBase:
    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.documents = [Document(**item) for item in raw]
        self.embedder = BGEEmbeddingProvider()
        self._texts = [self._searchable(document) for document in self.documents]
        self._bm25 = BM25Index(self._texts)
        self._embeddings = self.embedder.dense_many(self._texts)

    def search(self, query: str, limit: int = 5) -> ToolObservation:
        terms = normalize_terms(query)
        query_embedding = self.embedder.dense(query)
        dense_scores = [cosine_similarity(query_embedding, vector) for vector in self._embeddings]
        bm25_scores = self._bm25.scores(query)
        max_bm25 = max(bm25_scores, default=0.0) or 1.0
        default_dense_floor = 0.25 if self.embedder.backend.startswith("hashing-") else 0.55
        try:
            dense_floor = float(os.getenv("DENSE_MIN_SCORE", str(default_dense_floor)))
        except ValueError:
            dense_floor = default_dense_floor
        scored: list[SearchHit] = []
        for doc, dense_score, bm25_score in zip(self.documents, dense_scores, bm25_scores):
            if bm25_score <= 0 and dense_score < dense_floor:
                continue
            normalized_bm25 = bm25_score / max_bm25
            score = 0.55 * dense_score + 0.45 * normalized_bm25
            snippet = doc.content[:180] + ("…" if len(doc.content) > 180 else "")
            scored.append(SearchHit(
                doc.doc_id, doc.title, doc.version_date, doc.applicability, snippet, round(score, 4),
                page_start=doc.page_start,
                page_end=doc.page_end,
                chapter=doc.chapter,
                chunk_id=doc.chunk_id or doc.doc_id,
                parent_chunk_id=doc.parent_chunk_id,
                metadata={
                    "source_type": doc.source_type,
                    "image_refs": doc.image_refs,
                    "dense_score": round(dense_score, 4),
                    "bm25_score": round(bm25_score, 4),
                    "retrieval_routes": ["embedding", "bm25"],
                    "embedding_backend": self.embedder.backend,
                },
            ))
        scored.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return ToolObservation(
            tool="search_documents",
            ok=True,
            summary=f"Embedding+BM25 混合检索命中 {len(scored[:limit])} 份文档",
            data={
                "hits": [hit.__dict__ for hit in scored[:limit]],
                "query_terms": sorted(terms),
                "strategy": {
                    "dense_backend": self.embedder.backend,
                    "sparse_backend": "okapi-bm25",
                    "dense_min_score": dense_floor,
                    "fusion": "weighted(embedding=0.55,bm25=0.45)",
                },
            },
        )

    @staticmethod
    def _searchable(doc: Document) -> str:
        return " ".join([doc.title, doc.applicability, doc.chapter, doc.content, *doc.tags])

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

