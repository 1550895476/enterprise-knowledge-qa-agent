from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .decision import DecisionModel, OfflineDecisionModel, answer_stable_question
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

    def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentResult:
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
        _emit_progress(progress_callback, "analyze", "正在理解问题与对话上下文")
        stable_answer = answer_stable_question(question)
        action = (
            Action("finish", answer=stable_answer, reason="稳定计算问题由安全路由直接回答，不调用模型或知识库")
            if stable_answer
            else self.model.choose(state, observation)
        )

        for step in range(1, self.max_steps + 1):
            state["actions"].append(action.name)
            if action.name in {"ask", "finish"}:
                _emit_progress(progress_callback, "generate", f"第 {step} 步：证据核对完成，正在生成回答", step)
                normalized_action_gaps = _normalize_gaps(action.gaps)
                gaps.extend(normalized_action_gaps)
                status = "needs_confirmation" if action.name == "ask" else ("insufficient_evidence" if normalized_action_gaps else "completed")
                trace.append(TraceEvent(step, "decision", action.name, None, {}, "未调用工具", None, action.reason, self.model.source))
                answer = action.answer.strip() if action.answer else ""
                if not answer and action.name == "ask":
                    answer = (
                        "我找到了相关资料，但其中的内容或适用范围存在差异，暂时不能直接给出唯一答案。"
                        "请补充具体的产品型号、适用部门、制度版本或使用场景，我会继续核对。"
                        if citations
                        else
                        "目前的信息还不足以确认你要查询的具体范围。请补充产品型号、制度名称、适用部门"
                        "或相关上下文，我会根据补充信息继续查找。"
                    )
                elif not answer and action.name == "finish" and action.gaps:
                    answer = "未能在当前知识库中找到足以回答该问题的可靠资料。请补充相关资料或调整问题后重试。"
                elif not answer:
                    answer = "本次没有生成可用回答，请补充问题背景后重试。"
                return AgentResult(question, answer, citations, _unique(gaps), status, trace)

            if action.name == "search":
                params = {"query": action.query}
                _emit_progress(progress_callback, "search", f"第 {step} 步：正在检索相关资料", step, {"query": action.query})
                observation = self.tools.search(action.query)
                state["search_count"] += 1
                if observation.ok:
                    state["hits"] = observation.data.get("hits", [])
                    for stage in observation.data.get("pipeline_trace", []):
                        trace.append(TraceEvent(
                            step=step,
                            kind="retrieval_stage",
                            action=stage["action"],
                            tool=stage.get("tool"),
                            parameters=stage.get("parameters", {}),
                            result_summary=stage.get("result_summary", ""),
                            next_action=stage.get("next_action"),
                            decision_basis=stage.get("decision_basis", "按混合检索流水线继续"),
                            decision_source=stage.get("decision_source", "retrieval_pipeline"),
                            status=stage.get("status", "completed"),
                            duration_ms=stage.get("duration_ms"),
                        ))
                else:
                    state["tool_failures"] += 1
            elif action.name == "read":
                params = {"doc_id": action.doc_id}
                _emit_progress(progress_callback, "read", f"第 {step} 步：正在读取原文 {action.doc_id}", step, {"doc_id": action.doc_id})
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

            _emit_progress(
                progress_callback,
                "observe",
                f"第 {step} 步：{observation.summary}，正在判断下一步",
                step,
                {"tool": observation.tool, "ok": observation.ok},
            )
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
        _emit_progress(progress_callback, "limit", f"已达到最大执行步数 {self.max_steps}，任务未完成", self.max_steps)
        trace.append(TraceEvent(self.max_steps, "limit", action.name, None, {}, "未完成：达到步数上限", None, "保护性停止", self.model.source))
        return AgentResult(question, "本次任务未完成：已达到执行步数上限，请缩小问题范围或稍后重试。", citations, _unique(gaps), "incomplete", trace)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _normalize_gaps(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if value else []


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    stage: str,
    message: str,
    step: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if callback:
        callback({"stage": stage, "message": message, "step": step, "details": details or {}})

