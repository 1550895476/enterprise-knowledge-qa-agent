from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable


@dataclass
class ParsedPage:
    page_no: int
    text: str
    title: str = ""
    layout: list[dict[str, Any]] | None = None
    image_paths: list[str] | None = None


@dataclass
class StructuredChunk:
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
    tags: list[str]
    image_refs: list[str]
    layout: dict[str, Any]
    source_type: str = "pdf_manual"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PdfPageParser:
    """Extract pages and basic layout/image references when pypdf is installed."""

    def parse(self, pdf_path: str | Path, image_dir: str | Path | None = None) -> list[ParsedPage]:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PDF 解析需要安装 pypdf") from exc
        reader = PdfReader(str(pdf_path))
        pages = []
        image_root = Path(image_dir) if image_dir else None
        if image_root:
            image_root.mkdir(parents=True, exist_ok=True)
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            image_paths = []
            if image_root:
                try:
                    for image_index, image in enumerate(page.images, start=1):
                        suffix = Path(image.name).suffix or ".bin"
                        target = image_root / f"page-{index:03d}-image-{image_index:02d}{suffix}"
                        target.write_bytes(image.data)
                        image_paths.append(str(target))
                except Exception:
                    # Text extraction should still succeed when one embedded
                    # image uses an unsupported PDF codec.
                    pass
            pages.append(ParsedPage(index, text, self._title(text), [], image_paths))
        return pages

    def _title(self, text: str) -> str:
        for line in text.splitlines():
            if re.match(r"^\s*(\d+(?:\.\d+)*)\s+\S+", line):
                return line.strip()
        return ""


class StructureAwareChunker:
    """Build parent/child chunks while retaining page, heading and layout data."""

    def __init__(self, max_chars: int = 420, merge_shorter_than: int = 100):
        self.max_chars = max_chars
        self.merge_shorter_than = merge_shorter_than

    def chunk(
        self,
        pages: list[ParsedPage],
        doc_id: str,
        title: str,
        version_date: str,
        published_at: str,
        applicability: str,
        image_summarizer: Callable[[str], str] | None = None,
    ) -> list[StructuredChunk]:
        output: list[StructuredChunk] = []
        for page in pages:
            chapter = page.title or "未命名章节"
            parent_id = f"{doc_id}-P{page.page_no:03d}"
            content = page.text.strip()
            if not content:
                continue
            # Keep tables and nearby captions together. Layout coordinates are
            # preserved from upstream parsers when available.
            paragraphs = [part.strip() for part in re.split(r"\n{2,}", content) if part.strip()]
            children = self._split(paragraphs)
            for index, child in enumerate(children, start=1):
                image_refs = list(page.image_paths or [])
                if image_refs and image_summarizer:
                    child = f"{child}\n图片摘要：{image_summarizer(image_refs[0])}"
                output.append(StructuredChunk(
                    chunk_id=f"{parent_id}-C{index:02d}",
                    parent_chunk_id=parent_id,
                    doc_id=doc_id,
                    title=title,
                    version_date=version_date,
                    published_at=published_at,
                    applicability=applicability,
                    page_start=page.page_no,
                    page_end=page.page_no,
                    chapter=chapter,
                    content=child,
                    tags=self._tags(title, chapter, child),
                    image_refs=image_refs,
                    layout={"blocks": page.layout or [], "page_no": page.page_no},
                ))
        return output

    def _split(self, paragraphs: list[str]) -> list[str]:
        merged: list[str] = []
        for paragraph in paragraphs:
            if merged and len(merged[-1]) < self.merge_shorter_than:
                merged[-1] = f"{merged[-1]}\n{paragraph}"
            else:
                merged.append(paragraph)
        chunks: list[str] = []
        for paragraph in merged:
            for start in range(0, len(paragraph), self.max_chars):
                chunks.append(paragraph[start:start + self.max_chars])
        return chunks

    def _tags(self, title: str, chapter: str, content: str) -> list[str]:
        terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,6}", f"{title} {chapter} {content}")
        return list(dict.fromkeys(terms))[:12]

