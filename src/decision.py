from __future__ import annotations

import ast
import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .models import Action, ToolObservation
from .tools import normalize_terms


def answer_stable_question(question: str) -> str | None:
    """Return deterministic answers for safe calculations before any model/KB call."""
    expression = question.strip()
    for token in ("请问", "帮我计算", "计算", "等于几", "等于多少", "是多少", "结果是", "？", "?", "。"):
        expression = expression.replace(token, "")
    expression = expression.strip()
    if not expression or len(expression) > 80 or not re.fullmatch(r"[\d\s+\-*/().%]+", expression):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_arithmetic(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    rendered = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    return f"{expression} = {rendered}"


def _eval_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if abs(node.value) > 1_000_000_000:
            raise ValueError("number too large")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
        left = _eval_arithmetic(node.left)
        right = _eval_arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        else:
            result = left % right
        if abs(result) > 1_000_000_000_000:
            raise OverflowError("result too large")
        return result
    raise ValueError("unsupported expression")


class DecisionModel(ABC):
    source = "model"

    @abstractmethod
    def choose(self, state: dict[str, Any], observation: ToolObservation | None) -> Action:
        raise NotImplementedError

    def expand_query(self, query: str) -> str:
        """Generate a HyDE-style hypothetical answer with the active model."""
        return query


class OfflineDecisionModel(DecisionModel):
    """Deterministic local model used to make every path reproducible.

    It receives the same state and tool observations as the live model adapter. The
    decisions are intentionally made after observations, not in the orchestrator.
    """

    source = "offline_model"

    def expand_query(self, query: str) -> str:
        terms = "、".join(sorted(normalize_terms(query)))
        return f"{query}。可能涉及：{terms}。需要从适用文档中核对事实、版本、适用范围和原文依据。"

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
                return Action(
                    "finish",
                    answer=direct_answer,
                    reason="问题属于可安全计算的稳定通用知识，直接回答，无需检索知识库",
                )
            query = self._contextual_query(question, history)
            return Action("search", query=query, reason="先检索与当前问题及上下文相关的文档")

        if observation.tool in {"search_documents", "hybrid_search"}:
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

        if observation.tool in {"read_document", "read_chunk"}:
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
        conflicts = self._find_conflicts(readings)
        if conflicts:
            return Action(
                "ask",
                answer=self._format_conflict_answer(conflicts),
                gaps=["需要确认适用范围与制度优先级"],
                reason="原文中的关键数值或结论互相冲突，不能擅自择一",
            )
        answer = self._answer_from_readings(readings)
        return Action("finish", answer=answer, reason=reason)

    def _has_conflict(self, readings: list[dict[str, Any]]) -> bool:
        return bool(self._find_conflicts(readings))

    def _find_conflicts(self, readings: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
        evidence: dict[str, list[dict[str, str]]] = {}
        for item in readings:
            content = item["content"]
            for key in ("报销期限", "餐补标准"):
                found = self._extract_value(content, key)
                if found:
                    evidence.setdefault(key, []).append({
                        "doc_id": item.get("source_doc_id") or item["doc_id"],
                        "title": item.get("title", "未命名文档"),
                        "version_date": item.get("version_date", "未知版本"),
                        "value": found,
                    })
        return {
            key: items
            for key, items in evidence.items()
            if len({item["value"] for item in items}) > 1
        }

    def _format_conflict_answer(self, conflicts: dict[str, list[dict[str, str]]]) -> str:
        groups = []
        for fact, items in conflicts.items():
            statements = "；".join(
                f"{item['doc_id']}《{item['title']}》（版本日期 {item['version_date']}）规定 {item['value']}"
                for item in items
            )
            groups.append(f"{fact}存在冲突：{statements}")
        return (
            "资料存在冲突：" + "；".join(groups) +
            "。当前资料没有明确适用范围或优先级，请确认你适用的制度、组织范围或生效版本后再确定答案。"
        )

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

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        protocol: str = "chat_completions",
        reasoning_effort: str = "none",
        response_format: str = "json_object",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.protocol = protocol
        self.reasoning_effort = reasoning_effort
        self.response_format = response_format

    def expand_query(self, query: str) -> str:
        instruction = (
            "你是企业知识库检索的 HyDE 查询扩展器。根据用户问题生成一段简短的假设性答案，"
            "包含可能出现于制度或产品手册中的关键术语、适用范围和事实表达。"
            "只输出扩展文本，不引用不存在的文档，不输出 JSON。"
        )
        if self.protocol == "responses":
            payload: dict[str, Any] = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": query},
                ],
                "max_output_tokens": 300,
            }
            endpoint = "responses"
            if self.reasoning_effort != "none":
                payload["reasoning"] = {"effort": self.reasoning_effort}
        else:
            payload = {
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": query},
                ],
            }
            endpoint = "chat/completions"
            if self.reasoning_effort != "none":
                payload["reasoning_effort"] = self.reasoning_effort
        result = self._request_json(payload, endpoint)
        expanded = self._response_content(result).strip()
        if not expanded:
            raise RuntimeError("模型接口请求成功，但没有返回 HyDE 文本；请核对模型名称和接口协议")
        return expanded

    def choose(self, state: dict[str, Any], observation: ToolObservation | None) -> Action:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置，不能使用 openai 模型")
        instruction = (
            "你是企业知识问答 Agent 的决策模型。只输出 JSON。先判断问题是否需要企业知识库："
            "简单计算、语言解释、稳定且低风险的通用知识或日常交流，可以直接 action=finish，给出 answer，"
            "gaps=[]，reason 写明通用知识直答，不得伪造引用，也不要调用 search。"
            "涉及企业制度、内部流程、产品手册、上传文档、具体型号、时效信息、用户明确要求查资料，"
            "或你对答案不确定时，必须先 search，再根据观察 read 原文；不得依赖模型记忆直接回答。"
            "根据问题、历史、已执行动作和工具观察选择下一动作，action 只能是 search/read/ask/finish。"
            "资料冲突时必须 ask，不能猜测；知识库无资料时 finish 并写 gaps。JSON 字段："
            "action, query, doc_id, answer, gaps, reason。对于需要知识库的问题，answer 只能基于 readings 中"
            "已读取的原文，并标注文档 ID、页码、章节和 Chunk ID；没有 readings 时不得生成企业或产品事实答案。"
        )
        user_input = json.dumps({"state": state, "observation": observation.__dict__ if observation else None}, ensure_ascii=False)
        if self.protocol == "responses":
            payload: dict[str, Any] = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_input},
                ],
                "max_output_tokens": 2000,
            }
            endpoint = "responses"
            if self.reasoning_effort != "none":
                payload["reasoning"] = {"effort": self.reasoning_effort}
            if self.response_format == "json_object":
                payload["text"] = {"format": {"type": "json_object"}}
        else:
            payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_input},
                ],
            }
            endpoint = "chat/completions"
            if self.reasoning_effort != "none":
                payload["reasoning_effort"] = self.reasoning_effort
            if self.response_format == "json_object":
                payload["response_format"] = {"type": "json_object"}
        result = self._request_json(payload, endpoint)
        content = self._response_content(result)
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        if not content:
            raise RuntimeError("模型接口请求成功，但没有返回决策内容；请核对模型名称和接口协议")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("模型返回了内容，但不是 Agent 所需的 JSON；请启用 JSON Object 输出或更换兼容模型") from exc
        action_name = data.get("action")
        if action_name not in {"search", "read", "ask", "finish"}:
            raise RuntimeError(f"模型返回了无效动作：{action_name or '缺少 action'}")
        raw_gaps = data.get("gaps", [])
        if isinstance(raw_gaps, str):
            normalized_gaps = [raw_gaps] if raw_gaps.strip() else []
        elif isinstance(raw_gaps, list):
            normalized_gaps = [str(item) for item in raw_gaps if str(item).strip()]
        else:
            normalized_gaps = [str(raw_gaps)] if raw_gaps else []
        return Action(
            name=action_name, query=data.get("query", ""), doc_id=data.get("doc_id", ""),
            answer=data.get("answer", ""), gaps=normalized_gaps, reason=data.get("reason", "模型决策"),
        )

    def _request_json(self, payload: dict[str, Any], endpoint: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read().decode("utf-8").strip()
        if not raw:
            protocol_name = "Responses API" if self.protocol == "responses" else "Chat Completions"
            raise RuntimeError(
                f"模型接口返回了空响应。当前使用 {protocol_name}，请在模型配置中改用服务商支持的协议后重新测试"
            )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("模型接口返回的不是 JSON；请检查 Base URL 是否指向 OpenAI-compatible API 根路径") from exc
        if not isinstance(result, dict):
            raise RuntimeError("模型接口返回格式错误：响应必须是 JSON Object")
        return result

    def _response_content(self, payload: dict[str, Any]) -> str:
        if self.protocol == "chat_completions":
            try:
                return str(payload["choices"][0]["message"]["content"] or "")
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("Chat Completions 响应缺少 choices[0].message.content") from exc
        if payload.get("output_text"):
            return str(payload["output_text"])
        for item in payload.get("output", []):
            for content in item.get("content", []):
                value = content.get("text") or content.get("output_text")
                if value:
                    return str(value)
        raise RuntimeError("Responses API 响应缺少可解析的 output_text；该模型可能不支持 Responses API")

