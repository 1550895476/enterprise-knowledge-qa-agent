from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .storage import MilvusVectorStore
from .tools import normalize_terms


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    parent_chunk_id: str | None
    doc_id: str
    title: str
    version_date: str
    published_at: str
    applicability: str
    page_start: int
    page_end: int
    chapter: str
    content: str
    tags: list[str] = field(default_factory=list)
    image_refs: list[str] = field(default_factory=list)
    layout: dict[str, Any] = field(default_factory=dict)
    source_type: str = "pdf_manual"


@dataclass
class RetrievalResult:
    candidates: list[dict[str, Any]]
    query: str
    hyde_query: str
    mcp_count: int
    confidence: float
    conflicts: list[dict[str, Any]]
    strategy: dict[str, Any]


class BGEEmbeddingProvider:
    """BGE-M3 wrapper; a deterministic lexical vector is used without weights."""

    def __init__(self):
        self.backend = "lexical-fallback"
        self.model = None
        try:
            import os
            if os.getenv("ENABLE_BGE", "0") == "1":
                from FlagEmbedding import BGEM3FlagModel  # type: ignore

                self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
                self.backend = "bge-m3"
        except Exception:
            self.model = None

    def dense(self, text: str) -> list[float]:
        if self.model:
            output = self.model.encode([text], batch_size=1, max_length=512)
            return list(output["dense_vecs"][0])
        vector = [0.0] * 32
        for token in normalize_terms(text):
            # Python's built-in hash is randomized per process; use a stable
            # digest so offline ranking and recorded traces are reproducible.
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % len(vector)
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class BGEReranker:
    """BGE-Reranker wrapper with lexical score fallback."""

    def __init__(self):
        self.backend = "lexical-fallback"
        self.model = None
        try:
            import os
            if os.getenv("ENABLE_BGE", "0") == "1":
                from FlagEmbedding import FlagReranker  # type: ignore

                self.model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
                self.backend = "bge-reranker"
        except Exception:
            self.model = None

    def score(self, query: str, text: str) -> float:
        if self.model:
            return float(self.model.compute_score([query, text]))
        query_terms = normalize_terms(query)
        text_terms = normalize_terms(text)
        return len(query_terms & text_terms) / max(len(query_terms), 1)


class HybridRetriever:
    def __init__(self, chunks_path: str | Path, store: MilvusVectorStore | None = None):
        raw = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in raw]
        self.store = store or MilvusVectorStore()
        self.embedder = BGEEmbeddingProvider()
        self.reranker = BGEReranker()
        self.store.upsert([self._row(chunk) for chunk in self.chunks])

    def retrieve(self, query: str, product_id: str | None = None, top_p: float = 0.85, limit: int = 8) -> RetrievalResult:
        scoped = [chunk for chunk in self.chunks if not product_id or product_id in chunk.doc_id]
        hyde_query = self._hyde(query)
        dense = self._rank(hyde_query, scoped, mode="dense")
        sparse = self._rank(query, scoped, mode="sparse")
        mcp = self._mcp_rank(query, scoped)
        fused = self._rrf_fuse([dense, sparse, mcp])
        for item in fused:
            item["rerank_score"] = round(self.reranker.score(query, item["content"] + " " + item["chapter"]), 4)
            item["trust_score"] = self._trust_score(item)
            item["final_score"] = round(0.6 * item["rerank_score"] + 0.4 * item["trust_score"], 4)
        fused.sort(key=lambda item: (-item["final_score"], item["chunk_id"]))
        selected = self._top_p(fused[:limit], top_p)
        return RetrievalResult(
            candidates=selected,
            query=query,
            hyde_query=hyde_query,
            mcp_count=len(mcp),
            confidence=round(selected[0]["final_score"] if selected else 0.0, 4),
            conflicts=self._detect_conflicts(selected),
            strategy={
                "dense_backend": self.embedder.backend,
                "reranker_backend": self.reranker.backend,
                "fusion": "RRF(k=60)",
                "top_p": top_p,
                "metadata_rank": "version_date + published_at + applicability",
            },
        )

    def _rank(self, query: str, chunks: list[Chunk], mode: str) -> list[dict[str, Any]]:
        query_terms = normalize_terms(query)
        query_vector = self.embedder.dense(query)
        scored = []
        for chunk in chunks:
            text = " ".join([chunk.title, chunk.applicability, chunk.chapter, chunk.content, *chunk.tags])
            sparse_score = len({term for term in query_terms if term in text.lower()}) / max(len(query_terms), 1)
            dense_score = self._cosine(query_vector, self.embedder.dense(text))
            score = dense_score if mode == "dense" else sparse_score
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [self._candidate(chunk, score, mode) for score, chunk in scored]

    def _mcp_rank(self, query: str, chunks: list[Chunk]) -> list[dict[str, Any]]:
        # MCP retrieval is represented as a separately attributed route. In a
        # production deployment this method is replaced by the MCP connector.
        return self._rank(query + " 产品手册", chunks, mode="sparse")

    def _rrf_fuse(self, lists: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for ranked in lists:
            for rank, item in enumerate(ranked, start=1):
                entry = merged.setdefault(item["chunk_id"], dict(item))
                entry["rrf_score"] = entry.get("rrf_score", 0.0) + 1.0 / (k + rank)
                entry["retrieval_routes"] = sorted(set(entry.get("retrieval_routes", [])) | {item["route"]})
        return list(merged.values())

    def _top_p(self, ranked: list[dict[str, Any]], top_p: float) -> list[dict[str, Any]]:
        if not ranked:
            return []
        positive = [max(item["final_score"], 0.001) for item in ranked]
        total = sum(positive)
        selected: list[dict[str, Any]] = []
        cumulative = 0.0
        for item, score in zip(ranked, positive):
            selected.append(item)
            cumulative += score / total
            if cumulative >= top_p:
                break
        return selected

    def _trust_score(self, item: dict[str, Any]) -> float:
        def parse(value: str) -> datetime:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime(1970, 1, 1)

        # Metadata affects ranking but cannot override a direct content conflict.
        age_days = max(0, (datetime.now() - parse(item["published_at"])).days)
        recency = max(0.0, 1.0 - min(age_days, 3650) / 3650)
        return round(0.65 * recency + 0.35 * (1.0 if item["applicability"] else 0.5), 4)

    def _detect_conflicts(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in candidates:
            key = self._fact_key(item["content"])
            if key:
                groups.setdefault(key, []).append(item)
        conflicts = []
        for key, items in groups.items():
            values = {self._fact_value(item["content"]) for item in items}
            if len(values) > 1:
                conflicts.append({"fact": key, "values": sorted(values), "chunk_ids": [item["chunk_id"] for item in items]})
        return conflicts

    def _fact_key(self, text: str) -> str | None:
        if "电源键" in text and ("秒" in text or "侧面" in text or "顶部" in text):
            return "开机步骤"
        return None

    def _fact_value(self, text: str) -> str:
        match = re.search(r"(侧面|顶部|背面).{0,8}(\d+)秒", text)
        return "".join(match.groups()) if match else text[:40]

    def _hyde(self, query: str) -> str:
        terms = sorted(normalize_terms(query))
        return f"{query} 可能涉及：{'、'.join(terms)}。请在产品手册中核对操作步骤、适用型号和页码。"

    def _candidate(self, chunk: Chunk, score: float, route: str) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "parent_chunk_id": chunk.parent_chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "version_date": chunk.version_date,
            "published_at": chunk.published_at,
            "applicability": chunk.applicability,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chapter": chunk.chapter,
            "content": chunk.content,
            "image_refs": chunk.image_refs,
            "layout": chunk.layout,
            "source_type": chunk.source_type,
            "initial_score": round(score, 4),
            "route": route,
        }

    def _row(self, chunk: Chunk) -> dict[str, Any]:
        return {
            "id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.content,
            "metadata": {
                "version_date": chunk.version_date,
                "published_at": chunk.published_at,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chapter": chunk.chapter,
                "parent_chunk_id": chunk.parent_chunk_id,
                "image_refs": chunk.image_refs,
            },
        }

    def _cosine(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

