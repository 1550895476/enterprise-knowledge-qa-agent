from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - keeps the offline CLI dependency-free
    FastAPI = None
    BaseModel = object

from .advanced_agent import EnterpriseKnowledgeAgent
from .document_service import DocumentUploadError, DocumentUploadService
from .runtime_config import ModelConfigError, RuntimeModelConfig
from pathlib import Path


if FastAPI is not None:
    app = FastAPI(title="Enterprise Knowledge QA Agent", version="0.4.0")
    agent = EnterpriseKnowledgeAgent()
    document_service = DocumentUploadService(agent.retriever)
    model_config = RuntimeModelConfig()
    static_root = Path(__file__).resolve().parent / "static"

    class AskRequest(BaseModel):
        question: str = Field(min_length=1)
        history: list[dict[str, str]] = Field(default_factory=list)
        product_choice: str | None = None
        document_ids: list[str] = Field(default_factory=list)

    class ModelConfigRequest(BaseModel):
        provider: str = "offline"
        model: str = "gpt-4o-mini"
        base_url: str = "https://api.openai.com/v1"
        api_key: str | None = None

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return FileResponse(static_root / "index.html", media_type="text/html")

    @app.get("/config/model")
    def get_model_config() -> dict[str, Any]:
        return model_config.public()

    @app.put("/config/model")
    def update_model_config(request: ModelConfigRequest) -> dict[str, Any]:
        try:
            return model_config.update(request.provider, request.model, request.base_url, request.api_key)
        except ModelConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/config/model/test")
    def test_model_config(request: ModelConfigRequest) -> dict[str, Any]:
        try:
            return model_config.test_connection(request.provider, request.model, request.base_url, request.api_key)
        except ModelConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
                decision_model=model_config.build_model(),
            ).to_dict()
        except DocumentUploadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
else:
    app = None

