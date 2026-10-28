from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ActionName = Literal["search", "read", "ask", "finish"]


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    version_date: str
    applicability: str
    content: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    version_date: str
    applicability: str
    snippet: str
    score: float


@dataclass(frozen=True)
class ToolObservation:
    tool: str
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    name: ActionName
    query: str = ""
    doc_id: str = ""
    reason: str = ""
    answer: str = ""
    gaps: list[str] = field(default_factory=list)


@dataclass
class TraceEvent:
    step: int
    kind: str
    action: str
    tool: str | None
    parameters: dict[str, Any]
    result_summary: str
    next_action: str | None
    decision_basis: str
    decision_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    gaps: list[str]
    status: str
    trace: list[TraceEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "gaps": self.gaps,
            "status": self.status,
            "trace": [event.to_dict() for event in self.trace],
        }

