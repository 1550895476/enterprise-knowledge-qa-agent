from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .models import Action, ToolObservation


class DecisionModel(ABC):
    source = "model"

    @abstractmethod
    def choose(self, state: dict[str, Any], observation: ToolObservation | None) -> Action:
        raise NotImplementedError


class OfflineDecisionModel(DecisionModel):
    """Deterministic local model used to make every path reproducible.

    It receives the same state and tool observations as the live model adapter. The
    decisions are intentionally made after observations, not in the orchestrator.
    """

    source = "offline_model"

    def choose(self, state: dict[str, Any], observation: ToolObservation | None) -> Action:
        question = state["question"]
        history = state.get("history", [])
        search_count = state.get("search_count", 0)
        read_count = state.get("read_count", 0)
        failures = state.get("tool_failures", 0)
        hits = state.get("hits", [])
        readings = state.get("readings", [])

        if observation is None:
            query = self._contextual_query(question, history)
            return Action("search", query=query, reason="先检索与当前问题及上下文相关的文档")

        if observation.tool == "search_documents":
            if not observation.ok:
                if search_count < 2:
                    return Action("search", query=self._broaden(question), reason="检索工具可重试，改写查询后再次检索")
                return Action("finish", answer="检索服务连续失败，暂时无法核实制度原文。", gaps=["检索工具不可用"], reason="达到检索重试上限")
            if not hits:
                if search_count < 2:
                    return Action("search", query=self._broaden(question), reason="首轮无命中，去除限定词并补充同义词检索")
                return Action("finish", answer="当前知识库没有找到足以回答该问题的资料。", gaps=["缺少覆盖该问题的制度或 FAQ"], reason="补充检索后仍无可靠资料")
            unread = [hit["doc_id"] for hit in hits if hit["doc_id"] not in {item["doc_id"] for item in readings}]
            if unread and read_count < 3:
                return Action("read", doc_id=unread[0], reason="检索摘要不足以核对适用范围和原文，读取首个候选")
            return self._finalize(state, reason="已有足够原文证据")

        if observation.tool == "read_document":
            if not observation.ok:
                if failures < 2 and hits:
                    unread = [hit["doc_id"] for hit in hits if hit["doc_id"] not in {item["doc_id"] for item in readings}]
                    if unread:
                        return Action("read", doc_id=unread[0], reason="当前文档读取失败，改读另一份候选文档")
                return Action("finish", answer="文档读取失败，无法核验原文。", gaps=["文档读取工具失败"], reason="没有可核验的原文")
            unread = [hit["doc_id"] for hit in hits if hit["doc_id"] not in {item["doc_id"] for item in readings}]
            if unread and self._needs_cross_check(state):
                return Action("read", doc_id=unread[0], reason="已发现多个可能适用的文档，继续读取以检查冲突")
            return self._finalize(state, reason="读取结果已足以回答或说明资料缺口")

        return self._finalize(state, reason="工具观察已完成")

    def _contextual_query(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question
        previous = " ".join(item.get("content", "") for item in history[-4:])
        # Resolve a scoped follow-up against the latest subject instead of
        # carrying the previous employee scope as a competing scope.
        if "研发" in question or "开发" in question:
            previous = previous.replace("普通员工", "")
        elif "普通" in question:
            previous = previous.replace("研发", "").replace("开发", "")
        return f"{previous} {question}"

    def _broaden(self, question: str) -> str:
        # Keep the retry semantically narrow. Generic words such as “制度”
        # occur in every document and would turn an out-of-scope query into
        # false positives.
        return f"{question} 相关"

    def _needs_cross_check(self, state: dict[str, Any]) -> bool:
        hits = state.get("hits", [])
        if not (len(hits) > len(state.get("readings", [])) and len(state.get("readings", [])) < 2):
            return False
        # Read a second source when the top candidates are similarly relevant.
        # A strongly scoped hit (e.g. “研发员工”) should not be made
        # ambiguous merely because a broader FAQ also mentions the topic.
        return len(hits) > 1 and (hits[0].get("score", 0) - hits[1].get("score", 0)) <= 0.15

    def _finalize(self, state: dict[str, Any], reason: str) -> Action:
        readings = state.get("readings", [])
        if not readings:
            return Action("finish", answer="没有可核验的原文，暂不能给出可靠答案。", gaps=["缺少原文证据"], reason=reason)
        if self._has_conflict(readings):
            return Action(
                "ask",
                answer="资料存在冲突：不同文档给出了不同规定，且当前资料没有明确适用范围或优先级。请确认你适用的制度、组织范围或生效版本后再确定答案。",
                gaps=["需要确认适用范围与制度优先级"],
                reason="原文中的关键数值或结论互相冲突，不能擅自择一",
            )
        answer = self._answer_from_readings(readings)
        return Action("finish", answer=answer, reason=reason)

    def _has_conflict(self, readings: list[dict[str, Any]]) -> bool:
        values: dict[str, set[str]] = {}
        for item in readings:
            content = item["content"]
            for key in ("报销期限", "餐补标准"):
                found = self._extract_value(content, key)
                if found:
                    values.setdefault(key, set()).add(found)
        return any(len(found) > 1 for found in values.values())

    def _extract_value(self, content: str, key: str) -> str | None:
        import re

        if key == "报销期限":
            match = re.search(r"(\d+)\s*天", content)
            return match.group(0) if match else None
        match = re.search(r"(\d+)\s*元/天", content)
        return match.group(0) if match else None

    def _answer_from_readings(self, readings: list[dict[str, Any]]) -> str:
        primary = readings[0]
        return f"根据 {primary['doc_id']}（{primary['title']}，版本日期 {primary['version_date']}），{primary['content']}"


class OpenAICompatibleDecisionModel(DecisionModel):
    source = "openai_compatible_model"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def choose(self, state: dict[str, Any], observation: ToolObservation | None) -> Action:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置，不能使用 openai 模型")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": (
                    "你是企业知识问答 Agent 的决策模型。只输出 JSON。根据问题、历史、"
                    "已执行动作和工具观察选择下一动作。action 只能是 search/read/ask/finish。"
                    "资料冲突时必须 ask，不能猜测；无资料时 finish 并写 gaps。JSON 字段："
                    "action, query, doc_id, answer, gaps, reason。"
                )},
                {"role": "user", "content": json.dumps({"state": state, "observation": observation.__dict__ if observation else None}, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(content)
        return Action(
            name=data["action"], query=data.get("query", ""), doc_id=data.get("doc_id", ""),
            answer=data.get("answer", ""), gaps=data.get("gaps", []), reason=data.get("reason", "模型决策"),
        )

