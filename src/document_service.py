from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ingestion import ParsedPage, PdfPageParser, StructureAwareChunker
from .retrieval import HybridRetriever


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentUploadError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    checksum_sha256: str
    status: str
    chunk_count: int
    created_at: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentUploadService:
    """Persist uploads, build chunks, and attach them to the live retriever."""

    def __init__(
        self,
        retriever: HybridRetriever,
        runtime_root: str | Path | None = None,
        max_upload_bytes: int | None = None,
    ):
        project_root = Path(__file__).resolve().parent.parent
        self.runtime_root = Path(runtime_root or project_root / "runtime" / "uploads")
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = max_upload_bytes or int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
        self.retriever = retriever
        self.parser = PdfPageParser()
        self.chunker = StructureAwareChunker()
        self._records: dict[str, DocumentRecord] = {}
        self._lock = threading.RLock()
        self._load_existing()

    def ingest_bytes(self, filename: str, payload: bytes, content_type: str = "application/octet-stream") -> DocumentRecord:
        safe_name = Path(filename or "").name
        extension = Path(safe_name).suffix.lower()
        if not safe_name or extension not in SUPPORTED_EXTENSIONS:
            raise DocumentUploadError("仅支持 PDF、TXT 和 Markdown 文件", 415)
        if not payload:
            raise DocumentUploadError("上传文件为空")
        if len(payload) > self.max_upload_bytes:
            raise DocumentUploadError(f"文件超过大小限制 {self.max_upload_bytes} bytes", 413)
        if extension == ".pdf" and not payload.startswith(b"%PDF"):
            raise DocumentUploadError("文件扩展名为 PDF，但内容不是有效 PDF")

        checksum = hashlib.sha256(payload).hexdigest()
        document_id = f"UPL-{checksum[:16].upper()}"
        with self._lock:
            existing = self._records.get(document_id)
            if existing and existing.status == "ready":
                return existing
            if existing and existing.status == "processing":
                raise DocumentUploadError("相同文件正在解析和索引，请稍后查询状态", 409)
            directory = self.runtime_root / document_id
            directory.mkdir(parents=True, exist_ok=True)
            source_path = directory / safe_name
            source_path.write_bytes(payload)
            record = DocumentRecord(
                document_id=document_id,
                filename=safe_name,
                content_type=content_type,
                size_bytes=len(payload),
                checksum_sha256=checksum,
                status="processing",
                chunk_count=0,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[document_id] = record
            self._persist_record(directory, record)

        try:
            pages = self._parse(source_path, extension)
            today = datetime.now(timezone.utc).date().isoformat()
            structured = self.chunker.chunk(
                pages=pages,
                doc_id=document_id,
                title=safe_name,
                version_date=today,
                published_at=today,
                applicability="用户上传文档",
            )
            chunks = []
            for chunk in structured:
                item = chunk.to_dict()
                item["source_type"] = f"user_upload_{extension.lstrip('.')}"
                chunks.append(item)
            if not chunks:
                raise DocumentUploadError("文件中没有可索引的文本内容", 422)
            self.retriever.add_chunks(chunks)
            (directory / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
            with self._lock:
                record.status = "ready"
                record.chunk_count = len(chunks)
                self._persist_record(directory, record)
            return record
        except DocumentUploadError as exc:
            self._mark_failed(document_id, str(exc))
            raise
        except Exception as exc:
            self._mark_failed(document_id, str(exc))
            raise DocumentUploadError(f"文档解析或索引失败：{exc}", 422) from exc

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(document_id)

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)

    def require_ready(self, document_ids: list[str]) -> None:
        for document_id in document_ids:
            record = self.get(document_id)
            if record is None:
                raise DocumentUploadError(f"上传文档不存在：{document_id}", 404)
            if record.status != "ready":
                raise DocumentUploadError(f"文档尚未完成索引：{document_id}，当前状态 {record.status}", 409)

    def _parse(self, source_path: Path, extension: str) -> list[ParsedPage]:
        if extension == ".pdf":
            return self.parser.parse(source_path, image_dir=source_path.parent / "images")
        raw = source_path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise DocumentUploadError("文本文件必须使用 UTF-8 或 GB18030 编码") from exc
        title = self._first_heading(text) or source_path.stem
        return [ParsedPage(1, text, title, [], [])]

    def _first_heading(self, text: str) -> str:
        for line in text.splitlines():
            value = line.strip().lstrip("#").strip()
            if value:
                return value[:100]
        return ""

    def _load_existing(self) -> None:
        for record_path in self.runtime_root.glob("*/record.json"):
            try:
                record = DocumentRecord(**json.loads(record_path.read_text(encoding="utf-8")))
                if record.status == "processing":
                    record.status = "failed"
                    record.error = "服务重启中断了上一次索引，请重新上传"
                    self._persist_record(record_path.parent, record)
                self._records[record.document_id] = record
                chunks_path = record_path.parent / "chunks.json"
                if record.status == "ready" and chunks_path.exists():
                    self.retriever.add_chunks(json.loads(chunks_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

    def _mark_failed(self, document_id: str, error: str) -> None:
        with self._lock:
            record = self._records[document_id]
            record.status = "failed"
            record.error = error
            self._persist_record(self.runtime_root / document_id, record)

    def _persist_record(self, directory: Path, record: DocumentRecord) -> None:
        target = directory / "record.json"
        temporary = directory / "record.json.tmp"
        temporary.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

