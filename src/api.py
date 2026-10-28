from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - keeps the offline CLI dependency-free
    FastAPI = None
    BaseModel = object

from .advanced_agent import EnterpriseKnowledgeAgent
from .document_service import DocumentUploadError, DocumentUploadService


if FastAPI is not None:
    app = FastAPI(title="Enterprise Knowledge QA Agent", version="0.3.0")
    agent = EnterpriseKnowledgeAgent()
    document_service = DocumentUploadService(agent.retriever)

    class AskRequest(BaseModel):
        question: str = Field(min_length=1)
        history: list[dict[str, str]] = Field(default_factory=list)
        product_choice: str | None = None
        document_ids: list[str] = Field(default_factory=list)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "retrieval": "hybrid",
            "graph": "langgraph-or-fallback",
            "milvus": agent.retriever.store.backend,
            "mongo": agent.trace_store.backend,
            "upload_types": ["pdf", "txt", "md"],
            "max_upload_bytes": document_service.max_upload_bytes,
        }

    @app.post("/documents/upload", status_code=201)
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            payload = await file.read(document_service.max_upload_bytes + 1)
            record = document_service.ingest_bytes(
                filename=file.filename or "",
                payload=payload,
                content_type=file.content_type or "application/octet-stream",
            )
            return record.to_dict()
        except DocumentUploadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        finally:
            await file.close()

    @app.get("/documents")
    def list_documents() -> dict[str, Any]:
        return {"documents": [record.to_dict() for record in document_service.list()]}

    @app.get("/documents/{document_id}/status")
    def document_status(document_id: str) -> dict[str, Any]:
        record = document_service.get(document_id)
        if record is None:
            raise HTTPException(status_code=404, detail="上传文档不存在")
        return record.to_dict()

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        try:
            if request.document_ids:
                document_service.require_ready(request.document_ids)
            return agent.run(
                request.question,
                request.history,
                request.product_choice,
                document_ids=request.document_ids,
            ).to_dict()
        except DocumentUploadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
else:
    app = None

