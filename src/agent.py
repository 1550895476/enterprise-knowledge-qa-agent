from __future__ import annotations

from pathlib import Path
from typing import Any

from .decision import DecisionModel, OfflineDecisionModel
from .models import Action, AgentResult, TraceEvent, ToolObservation
from .tools import FaultInjectingTools, KnowledgeBase


DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "documents.json"


class KnowledgeQAAgent:
    def __init__(self, tools: FaultInjectingTools, model: DecisionModel | None = None, max_steps: int = 6):
        self.tools = tools
        self.model = model or OfflineDecisionModel()
        self.max_steps = max_steps

    @classmethod
    def from_default_data(cls, model: DecisionModel | None = None, max_steps: int = 6, **faults: bool):
        return cls(FaultInjectingTools(KnowledgeBase(DEFAULT_DATA), **faults), model=model, max_steps=max_steps)

    def run(self, question: str, history: list[dict[str, str]] | None = None) -> AgentResult:
        state: dict[str, Any] = {
            "question": question,
            "history": history or [],
            "hits": [],
            "readings": [],
            "search_count": 0,
            "read_count": 0,
            "tool_failures": 0,
            "actions": [],
        }
        trace: list[TraceEvent] = []
        citations: list[dict[str, Any]] = []
        gaps: list[str] = []
        observation: ToolObservation | None = None
        action = self.model.choose(state, observation)

        for step in range(1, self.max_steps + 1):
            state["actions"].append(action.name)
            if action.name in {"ask", "finish"}:
                gaps.extend(action.gaps)
                status = "needs_confirmation" if action.name == "ask" else ("insufficient_evidence" if action.gaps else "completed")
                trace.append(TraceEvent(step, "decision", action.name, None, {}, "未调用工具", None, action.reason, self.model.source))
                return AgentResult(question, action.answer, citations, _unique(gaps), status, trace)

            if action.name == "search":
                params = {"query": action.query}
                observation = self.tools.search(action.query)
                state["search_count"] += 1
                if observation.ok:
                    state["hits"] = observation.data.get("hits", [])
                else:
                    state["tool_failures"] += 1
            elif action.name == "read":
                params = {"doc_id": action.doc_id}
                observation = self.tools.read(action.doc_id)
                state["read_count"] += 1
                if observation.ok:
                    item = observation.data
                    if item["doc_id"] not in {reading["doc_id"] for reading in state["readings"]}:
                        state["readings"].append(item)
                        citations.append(item.copy())
                else:
                    state["tool_failures"] += 1
            else:
                params = {}
                observation = ToolObservation("unknown", False, f"不支持的动作 {action.name}")

            next_action = self.model.choose(state, observation)
            trace.append(TraceEvent(
                step=step,
                kind="tool_call",
                action=action.name,
                tool=observation.tool,
                parameters=params,
                result_summary=observation.summary,
                next_action=next_action.name,
                decision_basis=next_action.reason,
                decision_source=self.model.source,
            ))
            action = next_action

        gaps.append(f"达到最大执行步数上限 {self.max_steps}")
        trace.append(TraceEvent(self.max_steps, "limit", action.name, None, {}, "未完成：达到步数上限", None, "保护性停止", self.model.source))
        return AgentResult(question, "本次任务未完成：已达到执行步数上限，请缩小问题范围或稍后重试。", citations, _unique(gaps), "incomplete", trace)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))

