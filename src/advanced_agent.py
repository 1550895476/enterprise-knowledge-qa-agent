from __future__ import annotations

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
        self.document_ids: list[str] | None = None
        self._chunks: dict[str, dict[str, Any]] = {}
        self._sync_chunks()

    def _sync_chunks(self) -> None:
        for chunk in self.retriever.chunks:
            if chunk.chunk_id in self._chunks:
                continue
            self._chunks[chunk.chunk_id] = {
                "doc_id": chunk.chunk_id,
                "source_doc_id": chunk.doc_id,
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

    def set_scope(self, product_id: str | None = None, document_ids: list[str] | None = None) -> None:
        self.product_id = product_id
        self.document_ids = document_ids

    def search(self, query: str) -> ToolObservation:
        self._sync_chunks()
        result = self.retriever.retrieve(
            query,
            product_id=self.product_id,
            document_ids=self.document_ids,
        )
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
        self._sync_chunks()
        item = self._chunks.get(doc_id)
        if not item:
            return ToolObservation("read_chunk", False, f"Chunk {doc_id} 不存在", {"doc_id": doc_id})
        return ToolObservation("read_chunk", True, f"已读取 {doc_id} 原文片段", item)


class EnterpriseState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    selected_product_id: str | None
    document_ids: list[str]
    product_id: str | None
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

    def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        selected_product_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> AgentResult:
        state: EnterpriseState = {
            "question": question,
            "history": history or [],
            "selected_product_id": selected_product_id,
            "document_ids": document_ids or [],
        }
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
            return _FallbackGraph(self._resolve_product, self._route_after_resolution, self._retrieve_answer, self._stop)

    def _resolve_product(self, state: EnterpriseState) -> EnterpriseState:
        document_ids = state.get("document_ids", [])
        if document_ids:
            resolution = ProductResolution(
                "not_applicable",
                state["question"],
                [],
                "用户明确指定了上传文档，按 document_ids 限定检索范围",
            )
        else:
            resolution = self.matcher.resolve(state["question"], state.get("history", []), state.get("selected_product_id"))
        state["resolution"] = {
            "status": resolution.status,
            "rewritten_query": resolution.rewritten_query,
            "candidates": resolution.candidates,
            "reason": resolution.reason,
        }
        if resolution.status == "resolved":
            state["product_id"] = resolution.candidates[0]["product_id"]
        return state

    def _route_after_resolution(self, state: EnterpriseState) -> str:
        return "retrieve" if state["resolution"]["status"] in {"resolved", "not_applicable"} else "stop"

    def _retrieve_answer(self, state: EnterpriseState) -> EnterpriseState:
        resolution = state["resolution"]
        if resolution["status"] == "ambiguous":
            return state
        # Scope is request-local. This prevents concurrent LangGraph runs from
        # leaking a product or uploaded-document filter into another request.
        scoped_tools = AdvancedTools(self.retriever)
        scoped_tools.set_scope(
            product_id=state.get("product_id"),
            document_ids=state.get("document_ids") or None,
        )
        basic = KnowledgeQAAgent(scoped_tools, max_steps=self.max_steps)
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
    def __init__(self, resolve, route, retrieve, stop=None):
        self.resolve, self.route, self.retrieve, self.stop = resolve, route, retrieve, stop

    def invoke(self, state: EnterpriseState) -> EnterpriseState:
        state = self.resolve(state)
        if self.route(state) == "retrieve":
            state = self.retrieve(state)
        else:
            # Mirror the LangGraph stop node in the dependency-free fallback.
            state = self.stop(state) if self.stop else state
        return state

