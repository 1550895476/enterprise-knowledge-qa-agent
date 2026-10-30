from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .search_backends import BM25Index, BGEEmbeddingProvider, cosine_similarity, normalize_terms
from .storage import MilvusVectorStore


_BGE_RERANKER_MODEL = None
_BGE_RERANKER_LOCK = threading.Lock()


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


class BGEReranker:
    """BGE-Reranker wrapper with lexical score fallback."""

    def __init__(self):
        global _BGE_RERANKER_MODEL
        self.backend = "lexical-fallback"
        self.model = None
        if os.getenv("ENABLE_BGE", "0") == "1":
            try:
                from FlagEmbedding import FlagReranker  # type: ignore

                with _BGE_RERANKER_LOCK:
                    if _BGE_RERANKER_MODEL is None:
                        _BGE_RERANKER_MODEL = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
                self.model = _BGE_RERANKER_MODEL
                self.backend = "bge-reranker"
            except Exception as exc:
                raise RuntimeError(
                    "ENABLE_BGE=1，但 BGE-Reranker 加载失败；请检查 requirements-ml.txt、模型缓存和网络"
                ) from exc

    def score(self, query: str, text: str) -> float:
        if self.model:
            value: Any = self.model.compute_score([query, text], normalize=True)
            while isinstance(value, (list, tuple)):
                if not value:
                    return 0.0
                value = value[0]
            return float(value)
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
        self._embedding_cache: dict[str, list[float]] = {}
        self._cache_embeddings(self.chunks)
        self.store.upsert([self._row(chunk) for chunk in self.chunks])

    def add_chunks(self, chunks: list[Chunk | dict[str, Any]]) -> int:
        """Add newly ingested chunks to the live index and backing store."""
        prepared = [item if isinstance(item, Chunk) else Chunk(**item) for item in chunks]
        with self._lock:
            existing = {chunk.chunk_id for chunk in self.chunks}
            added = [chunk for chunk in prepared if chunk.chunk_id not in existing]
            self.chunks.extend(added)
            if added:
                self._cache_embeddings(added)
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
        query_expansion_backend: str = "active_model",
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

        self._emit_progress(progress_callback, "sparse_retrieve", "正在使用原始问题执行 BM25 稀疏检索", {"query": query})
        started = time.perf_counter()
        sparse = self._rank(query, scoped, mode="sparse", route="sparse")
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "sparse_retrieve",
            "bm25_search",
            {"query": query},
            f"BM25 稀疏检索完成，命中 {len(sparse)} 个候选",
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
                if hyde_generator is None:
                    raise ValueError("未提供当前在线模型的 HyDE 生成器")
                hyde_query = hyde_generator(query).strip()
                if not hyde_query:
                    raise ValueError("模型返回了空的 HyDE 内容")
            except Exception as exc:
                expansion_error = f"{type(exc).__name__}: {exc}"
                hyde_query = ""
                expansion_status = "failed"
        hyde_summary = (
            "HyDE 模型生成失败，本次跳过 HyDE 召回"
            if expansion_status == "failed"
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
        hyde_dense = self._rank(hyde_query, scoped, mode="dense", route="hyde") if hyde_query else []
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
        default_floor = 0.0 if self.reranker.backend == "lexical-fallback" else 0.1
        try:
            rerank_floor = float(os.getenv("RERANK_MIN_SCORE", str(default_floor)))
        except ValueError:
            rerank_floor = default_floor
        qualified = [item for item in fused if item["rerank_score"] > rerank_floor]
        selected = self._top_p(qualified[:limit], top_p)
        self._record_stage(
            pipeline_trace,
            progress_callback,
            "rerank",
            "bge_reranker",
            {"query": query, "top_p": top_p, "limit": limit, "min_score": rerank_floor},
            f"统一重排完成，相关性门槛过滤 {len(fused) - len(qualified)} 个，Top-p 选出 {len(selected)} 个候选",
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
        routes = ["dense", "sparse", *(["hyde"] if hyde_dense else [])]
        return RetrievalResult(
            candidates=selected,
            query=query,
            hyde_query=hyde_query,
            confidence=round(selected[0]["final_score"] if selected else 0.0, 4),
            conflicts=conflicts,
            strategy={
                "dense_backend": self.embedder.backend,
                "sparse_backend": "okapi-bm25",
                "reranker_backend": self.reranker.backend,
                "routes": routes,
                "query_expansion": query_expansion_backend,
                "fusion": "RRF(k=60)",
                "top_p": top_p,
                "rerank_min_score": rerank_floor,
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
        if mode == "sparse":
            texts = [self._searchable(chunk) for chunk in chunks]
            route_scores = BM25Index(texts).scores(query)
        else:
            query_vector = self.embedder.dense(query)
            route_scores = [
                cosine_similarity(query_vector, self._embedding_cache[chunk.chunk_id])
                for chunk in chunks
            ]
        scored: list[tuple[float, Chunk]] = []
        for chunk, score in zip(chunks, route_scores):
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
            "dense_vector": self._embedding_cache.get(chunk.chunk_id, []),
            "metadata": {
                "embedding_backend": self.embedder.backend,
                "version_date": chunk.version_date,
                "published_at": chunk.published_at,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chapter": chunk.chapter,
                "parent_chunk_id": chunk.parent_chunk_id,
                "image_refs": chunk.image_refs,
            },
        }

    @staticmethod
    def _searchable(chunk: Chunk) -> str:
        return " ".join([chunk.title, chunk.applicability, chunk.chapter, chunk.content, *chunk.tags])

    def _cache_embeddings(self, chunks: list[Chunk]) -> None:
        missing = [chunk for chunk in chunks if chunk.chunk_id not in self._embedding_cache]
        vectors = self.embedder.dense_many([self._searchable(chunk) for chunk in missing])
        self._embedding_cache.update({chunk.chunk_id: vector for chunk, vector in zip(missing, vectors)})

