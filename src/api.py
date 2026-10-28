from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - keeps the offline CLI dependency-free
    FastAPI = None
    BaseModel = object

from .advanced_agent import EnterpriseKnowledgeAgent


if FastAPI is not None:
    app = FastAPI(title="Enterprise Knowledge QA Agent", version="0.2.0")
    agent = EnterpriseKnowledgeAgent()

    class AskRequest(BaseModel):
        question: str = Field(min_length=1)
        history: list[dict[str, str]] = Field(default_factory=list)
        product_choice: str | None = None

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "retrieval": "hybrid", "graph": "langgraph-or-fallback", "milvus": agent.retriever.store.backend, "mongo": agent.trace_store.backend}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        try:
            return agent.run(request.question, request.history, request.product_choice).to_dict()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
else:
    app = None

