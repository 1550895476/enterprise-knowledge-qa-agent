from __future__ import annotations

import re
from typing import Any

from src.decision import DecisionModel, answer_stable_question
from src.models import Action, ToolObservation
from src.tools import normalize_terms


class DeterministicTestDecisionModel(DecisionModel):
    """Test-only decision double; it is never exposed by the application."""

    source = "deterministic_test_double"

    def expand_query(self, query: str) -> str:
        terms = "、".join(sorted(normalize_terms(query)))
        return f"{query}。可能涉及：{terms}。需要核对事实、版本、适用范围和原文依据。"

    def choose(self, state: dict[str, Any], observation: ToolObservation | None) -> Action:
        question = state["question"]
        history = state.get("history", [])
        search_count = state.get("search_count", 0)
        read_count = state.get("read_count", 0)
        failures = state.get("tool_failures", 0)
        hits = state.get("hits", [])
        readings = state.get("readings", [])

        if observation is None:
            direct_answer = answer_stable_question(question)
            if direct_answer:
                return Action("finish", answer=direct_answer, reason="测试桩直接处理稳定计算")
            return Action("search", query=self._contextual_query(question, history), reason="先检索相关资料")

        if observation.tool in {"search_documents", "hybrid_search"}:
            if not observation.ok:
                if search_count < 2:
                    return Action("search", query=f"{question} 相关", reason="检索失败后改写查询")
                return Action("finish", answer="检索服务连续失败。", gaps=["检索工具不可用"], reason="达到重试上限")
            if not hits:
                if search_count < 2:
                    return Action("search", query=f"{question} 相关", reason="首轮无命中，补充检索")
                return Action("finish", answer="当前知识库没有找到足以回答该问题的资料。", gaps=["缺少覆盖该问题的制度或 FAQ"], reason="补充检索后仍无资料")
            unread = [hit["doc_id"] for hit in hits if hit["doc_id"] not in {item["doc_id"] for item in readings}]
            if unread and read_count < 3:
                return Action("read", doc_id=unread[0], reason="读取首个候选原文")
            return self._finalize(state, "已有原文证据")

        if observation.tool in {"read_document", "read_chunk"}:
            if not observation.ok:
                unread = [hit["doc_id"] for hit in hits if hit["doc_id"] not in {item["doc_id"] for item in readings}]
                if failures < 2 and unread:
                    return Action("read", doc_id=unread[0], reason="当前读取失败，改读另一候选")
                return Action("finish", answer="文档读取失败。", gaps=["文档读取工具失败"], reason="无可核验原文")
            unread = [hit["doc_id"] for hit in hits if hit["doc_id"] not in {item["doc_id"] for item in readings}]
            if unread and self._needs_cross_check(state):
                return Action("read", doc_id=unread[0], reason="继续读取以检查冲突")
            return self._finalize(state, "读取结果足以决策")

        return self._finalize(state, "工具观察完成")

    def _contextual_query(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question
        previous = " ".join(item.get("content", "") for item in history[-4:])
        if "研发" in question or "开发" in question:
            previous = previous.replace("普通员工", "")
        elif "普通" in question:
            previous = previous.replace("研发", "").replace("开发", "")
        return f"{previous} {question}"

    def _needs_cross_check(self, state: dict[str, Any]) -> bool:
        hits = state.get("hits", [])
        readings = state.get("readings", [])
        return len(hits) > len(readings) and len(readings) < 2 and len(hits) > 1 and (hits[0].get("score", 0) - hits[1].get("score", 0)) <= 0.15

    def _finalize(self, state: dict[str, Any], reason: str) -> Action:
        readings = state.get("readings", [])
        if not readings:
            return Action("finish", answer="没有可核验的原文。", gaps=["缺少原文证据"], reason=reason)
        conflicts = self._find_conflicts(readings)
        if conflicts:
            groups = []
            for fact, items in conflicts.items():
                statements = "；".join(
                    f"{item['doc_id']}《{item['title']}》（版本日期 {item['version_date']}）规定 {item['value']}"
                    for item in items
                )
                groups.append(f"{fact}存在冲突：{statements}")
            return Action(
                "ask",
                answer=(
                    "资料存在冲突：" + "；".join(groups) +
                    "。当前资料没有明确适用范围或优先级，请确认适用制度后再确定答案。"
                ),
                gaps=["需要确认适用范围与制度优先级"],
                reason="原文结论冲突，不能擅自选择",
            )
        primary = readings[0]
        return Action(
            "finish",
            answer=f"根据 {primary['doc_id']}（{primary['title']}，版本日期 {primary['version_date']}），{primary['content']}",
            reason=reason,
        )

    def _find_conflicts(self, readings: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
        evidence: dict[str, list[dict[str, str]]] = {}
        for item in readings:
            for fact, pattern in (("报销期限", r"(\d+)\s*天"), ("餐补标准", r"(\d+)\s*元/天")):
                match = re.search(pattern, item["content"])
                if match:
                    evidence.setdefault(fact, []).append({
                        "doc_id": item.get("source_doc_id") or item["doc_id"],
                        "title": item.get("title", "未命名文档"),
                        "version_date": item.get("version_date", "未知版本"),
                        "value": match.group(0),
                    })
        return {fact: items for fact, items in evidence.items() if len({item["value"] for item in items}) > 1}
