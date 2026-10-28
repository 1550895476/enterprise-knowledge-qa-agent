from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from .agent import KnowledgeQAAgent
from .generation import GroundedAnswerGenerator
from .models import AgentResult, ToolObservation, TraceEvent
from .query_understanding import ProductMatcher, ProductResolution
from .retrieval import HybridRetriever
from .storage import MongoTraceStore


class AdvancedTools:
    """Expose hybrid retrieval as the same search/read contract as the basic Agent."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.product_id: str | None = None
        self._chunks: dict[str, dict[str, Any]] = {}
        for chunk in retriever.chunks:
            self._chunks[chunk.chunk_id] = {
                "doc_id": chunk.chunk_id,
                "title": chunk.title,
                "version_date": chunk.version_date,
                "applicability": chunk.applicability,
                "content": chunk.content,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chapter": chunk.chapter,
                "chunk_id": chunk.chunk_id,
                "parent_chunk_id": chunk.parent_chunk_id,
                "image_refs": chunk.image_refs,
                "source_type": chunk.source_type,
            }

    def search(self, query: str) -> ToolObservation:
        result = self.retriever.retrieve(query, product_id=self.product_id)
        hits = []
        for item in result.candidates:
            hits.append({
                "doc_id": item["chunk_id"],
                "title": item["title"],
                "version_date": item["version_date"],
                "applicability": item["applicability"],
                "snippet": item["content"][:180],
                "score": item["final_score"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "chapter": item["chapter"],
                "chunk_id": item["chunk_id"],
                "parent_chunk_id": item["parent_chunk_id"],
                "metadata": {
                    "published_at": item["published_at"],
                    "retrieval_routes": item["retrieval_routes"],
                    "rerank_score": item["rerank_score"],
                    "trust_score": item["trust_score"],
                    "conflicts": result.conflicts,
                },
            })
        return ToolObservation(
            "hybrid_search",
            True,
            f"混合检索命中 {len(hits)} 个 Chunk，路线=dense+sparse+HyDE+MCP，RRF 后 BGE-Reranker 精排",
            {"hits": hits, "retrieval": {"hyde_query": result.hyde_query, "confidence": result.confidence, "conflicts": result.conflicts, "strategy": result.strategy}},
        )

    def read(self, doc_id: str) -> ToolObservation:
        item = self._chunks.get(doc_id)
        if not item:
            return ToolObservation("read_chunk", False, f"Chunk {doc_id} 不存在", {"doc_id": doc_id})
        return ToolObservation("read_chunk", True, f"已读取 {doc_id} 原文片段", item)


class EnterpriseState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    selected_product_id: str | None
    resolution: dict[str, Any]
    result: dict[str, Any]


class EnterpriseKnowledgeAgent:
    """LangGraph entry point with product gating before any KB retrieval."""

    def __init__(self, chunks_path: str | Path | None = None, products_path: str | Path | None = None, max_steps: int = 6):
        root = Path(__file__).resolve().parent.parent
        self.matcher = ProductMatcher(products_path or root / "data" / "products.json")
        self.retriever = HybridRetriever(chunks_path or root / "data" / "chunks.json")
        self.tools = AdvancedTools(self.retriever)
        self.trace_store = MongoTraceStore()
        self.generator = GroundedAnswerGenerator()
        self.max_steps = max_steps
        self.graph = self._build_graph()

    def run(self, question: str, history: list[dict[str, str]] | None = None, selected_product_id: str | None = None) -> AgentResult:
        state: EnterpriseState = {"question": question, "history": history or [], "selected_product_id": selected_product_id}
        final = self.graph.invoke(state)
        result = self._from_dict(final["result"])
        self.trace_store.append({"question": question, "resolution": final.get("resolution", {}), "result": result.to_dict()})
        return result

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph  # type: ignore

            graph = StateGraph(EnterpriseState)
            graph.add_node("resolve_product", self._resolve_product)
            graph.add_node("retrieve_answer", self._retrieve_answer)
            graph.add_node("stop", self._stop)
            graph.set_entry_point("resolve_product")
            graph.add_conditional_edges("resolve_product", self._route_after_resolution, {"retrieve": "retrieve_answer", "stop": "stop"})
            graph.add_edge("retrieve_answer", "stop")
            graph.add_edge("stop", END)
            return graph.compile()
        except ImportError:
            return _FallbackGraph(self._resolve_product, self._route_after_resolution, self._retrieve_answer)

    def _resolve_product(self, state: EnterpriseState) -> EnterpriseState:
        resolution = self.matcher.resolve(state["question"], state.get("history", []), state.get("selected_product_id"))
        state["resolution"] = {
            "status": resolution.status,
            "rewritten_query": resolution.rewritten_query,
            "candidates": resolution.candidates,
            "reason": resolution.reason,
        }
        if resolution.status == "resolved":
            self.tools.product_id = resolution.candidates[0]["product_id"]
        return state

    def _route_after_resolution(self, state: EnterpriseState) -> str:
        return "retrieve" if state["resolution"]["status"] in {"resolved", "not_applicable"} else "stop"

    def _retrieve_answer(self, state: EnterpriseState) -> EnterpriseState:
        resolution = state["resolution"]
        if resolution["status"] == "ambiguous":
            return state
        basic = KnowledgeQAAgent(self.tools, max_steps=self.max_steps)
        result = basic.run(resolution["rewritten_query"], history=state.get("history", []))
        # Keep the public response tied to the user's original wording; the
        # rewritten query remains visible in the graph resolution metadata and
        # search Trace.
        result.question = state["question"]
        if result.status == "completed" and result.citations:
            result.answer = self.generator.generate(resolution["rewritten_query"], result.citations)
        state["result"] = result.to_dict()
        return state

    def _stop(self, state: EnterpriseState) -> EnterpriseState:
        if "result" in state:
            return state
        resolution = state["resolution"]
        if resolution["status"] == "ambiguous":
            names = "、".join(item["canonical_name"] for item in resolution["candidates"])
            answer = f"请先确认商品型号：{names}。不同型号的说明书不能混用。"
            gap = "需要用户选择标准商品型号"
        else:
            answer = "无法在标准商品库中确认该商品型号，因此不会执行知识库检索。请提供完整型号。"
            gap = "商品型号无法确认"
        state["result"] = AgentResult(
            state["question"], answer, [], [gap], "needs_confirmation", [
                TraceEvent(1, "decision", "ask", None, {"candidates": resolution["candidates"]}, "未执行知识库检索", None, resolution["reason"], "product_matcher")
            ]
        ).to_dict()
        return state

    def _from_dict(self, payload: dict[str, Any]) -> AgentResult:
        return AgentResult(
            question=payload["question"], answer=payload["answer"], citations=payload["citations"], gaps=payload["gaps"], status=payload["status"],
            trace=[TraceEvent(**event) for event in payload["trace"]],
        )


class _FallbackGraph:
    def __init__(self, resolve, route, retrieve):
        self.resolve, self.route, self.retrieve = resolve, route, retrieve

    def invoke(self, state: EnterpriseState) -> EnterpriseState:
        state = self.resolve(state)
        if self.route(state) == "retrieve":
            state = self.retrieve(state)
        else:
            # Mirror the LangGraph stop node in the dependency-free fallback.
            state = self._stop(state)
        return state

