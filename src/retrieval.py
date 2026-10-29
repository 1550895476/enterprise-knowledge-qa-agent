from __future__ import annotations

import json
import hashlib
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

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
    confidence: float
    conflicts: list[dict[str, Any]]
    strategy: dict[str, Any]
    pipeline_trace: list[dict[str, Any]] = field(default_factory=list)


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
        self._lock = threading.RLock()
        self.store.upsert([self._row(chunk) for chunk in self.chunks])

    def add_chunks(self, chunks: list[Chunk | dict[str, Any]]) -> int:
        """Add newly ingested chunks to the live index and backing store."""
        prepared = [item if isinstance(item, Chunk) else Chunk(**item) for item in chunks]
        with self._lock:
            existing = {chunk.chunk_id for chunk in self.chunks}
            added = [chunk for chunk in prepared if chunk.chunk_id not in existing]
            self.chunks.extend(added)
            if added:
                self.store.upsert([self._row(chunk) for chunk in added])
            return len(added)

    def retrieve(
        self,
        query: str,
        product_id: str | None = None,
        document_ids: list[str] | None = None,
        top_p: float = 0.85,
        limit: int = 8,
        hyde_query: str | None = None,
        query_expansion_backend: str = "offline_model",
        hyde_generator: Callable[[str], str] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> RetrievalResult:
        with self._lock:
            scoped = list(self.chunks)
        if document_ids:
            allowed = set(document_ids)
            scoped = [chunk for chunk in scoped if chunk.doc_id in allowed]
        elif product_id:
            scoped = [chunk for chunk in scoped if product_id in chunk.doc_id]

        pipeline_trace: list[dict[str, Any]] = []

        self._emit_progress(progress_callback, "dense_retrieve", "正在使用原始问题执行 Dense 稠密检索", {"query": query})
        started = time.perf_counter()
        dense = self._rank(query, scoped, mode="dense", route="dense")
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "dense_retrieve",
            "dense_vector_search",
            {"query": query},
            f"Dense 稠密检索完成，命中 {len(dense)} 个候选",
            started,
        )

        self._emit_progress(progress_callback, "sparse_retrieve", "正在使用原始问题执行 Sparse 关键词检索", {"query": query})
        started = time.perf_counter()
        sparse = self._rank(query, scoped, mode="sparse", route="sparse")
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "sparse_retrieve",
            "sparse_vector_search",
            {"query": query},
            f"Sparse 关键词检索完成，命中 {len(sparse)} 个候选",
            started,
        )

        self._emit_progress(
            progress_callback,
            "hyde_generate",
            f"正在使用当前模型生成 HyDE 假设答案（{query_expansion_backend}）",
            {"query": query, "model_source": query_expansion_backend},
        )
        started = time.perf_counter()
        expansion_status = "completed"
        expansion_error = ""
        if hyde_query is None:
            try:
                hyde_query = (hyde_generator(query) if hyde_generator else self._hyde(query)).strip()
                if not hyde_query:
                    raise ValueError("模型返回了空的 HyDE 内容")
            except Exception as exc:
                expansion_error = f"{type(exc).__name__}: {exc}"
                hyde_query = self._hyde(query)
                query_expansion_backend = f"{query_expansion_backend}:offline_fallback"
                expansion_status = "degraded"
        hyde_summary = (
            "HyDE 模型生成失败，已使用离线模板继续"
            if expansion_status == "degraded"
            else "HyDE 假设答案生成完成"
        )
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "hyde_generate",
            "active_model",
            {"query": query, "model_source": query_expansion_backend},
            hyde_summary,
            started,
            status=expansion_status,
            details={"hyde_query": hyde_query, "error": expansion_error},
        )

        self._emit_progress(progress_callback, "hyde_retrieve", "正在使用 HyDE 假设答案执行 Dense 稠密检索", {"hyde_query": hyde_query})
        started = time.perf_counter()
        hyde_dense = self._rank(hyde_query, scoped, mode="dense", route="hyde")
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "hyde_retrieve",
            "dense_vector_search",
            {"query": hyde_query},
            f"HyDE 稠密检索完成，命中 {len(hyde_dense)} 个候选",
            started,
        )

        self._emit_progress(progress_callback, "rrf_fusion", "正在对三路召回结果执行 RRF 去重融合")
        started = time.perf_counter()
        fused = self._rrf_fuse([dense, sparse, hyde_dense])
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "rrf_fusion",
            "rrf_fusion",
            {"routes": ["dense", "sparse", "hyde"], "k": 60},
            f"RRF 融合完成，去重后保留 {len(fused)} 个候选",
            started,
        )

        self._emit_progress(progress_callback, "rerank", "正在使用重排模型统一精排并按 Top-p 截取")
        started = time.perf_counter()
        for item in fused:
            item["rerank_score"] = round(self.reranker.score(query, item["content"] + " " + item["chapter"]), 4)
            item["trust_score"] = self._trust_score(item)
            item["final_score"] = round(0.6 * item["rerank_score"] + 0.4 * item["trust_score"], 4)
        fused.sort(key=lambda item: (-item["final_score"], item["chunk_id"]))
        selected = self._top_p(fused[:limit], top_p)
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "rerank",
            "bge_reranker",
            {"query": query, "top_p": top_p, "limit": limit},
            f"统一重排完成，Top-p 选出 {len(selected)} 个候选",
            started,
        )

        self._emit_progress(progress_callback, "conflict_check", "正在核对版本、适用范围与内容冲突")
        started = time.perf_counter()
        conflicts = self._detect_conflicts(selected)
        conflict_summary = f"冲突检查完成，发现 {len(conflicts)} 组冲突" if conflicts else "冲突检查完成，未发现证据冲突"
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "conflict_check",
            "metadata_conflict_detector",
            {"candidate_count": len(selected)},
            conflict_summary,
            started,
            details={"conflict_count": len(conflicts)},
        )

        for index, event in enumerate(pipeline_trace):
            event["next_action"] = pipeline_trace[index + 1]["action"] if index + 1 < len(pipeline_trace) else "observe_search_result"
        return RetrievalResult(
            candidates=selected,
            query=query,
            hyde_query=hyde_query,
            confidence=round(selected[0]["final_score"] if selected else 0.0, 4),
            conflicts=conflicts,
            strategy={
                "dense_backend": self.embedder.backend,
                "reranker_backend": self.reranker.backend,
                "routes": ["dense", "sparse", "hyde"],
                "query_expansion": query_expansion_backend,
                "fusion": "RRF(k=60)",
                "top_p": top_p,
                "metadata_rank": "version_date + published_at + applicability",
            },
            pipeline_trace=pipeline_trace,
        )

    def _emit_progress(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        stage: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if callback:
            callback({"stage": stage, "message": message, "step": None, "details": details or {}})

    def _record_stage(
        self,
        trace: list[dict[str, Any]],
        callback: Callable[[dict[str, Any]], None] | None,
        action: str,
        tool: str,
        parameters: dict[str, Any],
        summary: str,
        started: float,
        status: str = "completed",
        details: dict[str, Any] | None = None,
    ) -> None:
        duration_ms = max(0, round((time.perf_counter() - started) * 1000))
        event = {
            "action": action,
            "tool": tool,
            "parameters": parameters,
            "result_summary": summary,
            "next_action": None,
            "decision_basis": "混合检索流水线按已配置策略继续",
            "decision_source": "retrieval_pipeline",
            "status": status,
            "duration_ms": duration_ms,
        }
        trace.append(event)
        progress_details = {"tool": tool, "status": status, "duration_ms": duration_ms, **(details or {})}
        self._emit_progress(callback, action, summary, progress_details)

    def _rank(self, query: str, chunks: list[Chunk], mode: str, route: str | None = None) -> list[dict[str, Any]]:
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
        return [self._candidate(chunk, score, route or mode) for score, chunk in scored]

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

