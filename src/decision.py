from __future__ import annotations

import ast
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .http_utils import urlopen_with_retry
from .models import Action, ToolObservation


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
            "你是企业知识问答 Agent 的决策模型。必须输出一个完整 JSON 对象：首字符必须是 {，末字符必须是 }；"
            "不能只输出单词 JSON，不能输出解释、Markdown 或思维过程。示例："
            '{"action":"search","query":"密码重置","doc_id":"","answer":"","gaps":[],"reason":"需要核对企业资料"}。'
            "先判断问题是否需要企业知识库："
            "简单计算、语言解释、稳定且低风险的通用知识或日常交流，可以直接 action=finish，给出 answer，"
            "gaps=[]，reason 写明通用知识直答，不得伪造引用，也不要调用 search。"
            "涉及企业制度、内部流程、产品手册、上传文档、具体型号、时效信息、用户明确要求查资料，"
            "或你对答案不确定时，必须先 search，再根据观察 read 原文；不得依赖模型记忆直接回答。"
            "根据问题、历史、已执行动作和工具观察选择下一动作，action 只能是 search/read/ask/finish。"
            "search 返回的 hits 和 snippet 只能用于选择待读文档，不是可引用原文：只要 hits 非空且 readings 为空，"
            "下一动作必须是 read，并从 hits 选择 doc_id；禁止直接 ask 或基于摘要回答。可能存在版本或数值差异时，"
            "必须继续 read 足够的相关候选完成交叉核对，不能读完第一条就直接 finish，也不能在未读取冲突原文前 ask。"
            "如果已经执行了完整 search 且 hits 为空，说明知识库资料不足，应 finish、给出友好拒答并填写 gaps；"
            "除非用户问题本身缺少可用于检索的明确对象，否则不要改成 ask。"
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
        content = self._response_content(result).strip()
        if not content:
            raise RuntimeError("模型接口请求成功，但没有返回决策内容；请核对模型名称和接口协议")
        try:
            data = _extract_action_json(content)
        except ValueError as exc:
            raise RuntimeError(
                "模型返回了内容，但没有可解析的 Agent JSON；请启用 JSON Object 输出或更换兼容模型"
            ) from exc
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
        action = Action(
            name=action_name, query=data.get("query", ""), doc_id=data.get("doc_id", ""),
            answer=data.get("answer", ""), gaps=normalized_gaps, reason=data.get("reason", "模型决策"),
        )
        return _enforce_evidence_protocol(action, state, observation)

    def _request_json(self, payload: dict[str, Any], endpoint: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Enterprise-Knowledge-QA-Agent/1.0",
            },
            method="POST",
        )
        try:
            with urlopen_with_retry(request, timeout=45, attempts=3) as response:
                raw = response.read().decode("utf-8").strip()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                detail = ""
            suffix = f"：{detail}" if detail else ""
            raise RuntimeError(f"模型接口返回 HTTP {exc.code}{suffix}") from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(
                f"连接模型接口失败，已自动重试 3 次：{reason}。请稍后重试，并确认代理、VPN 或中转站状态"
            ) from exc
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


def _extract_action_json(content: str) -> dict[str, Any]:
    """Extract an action object from plain JSON, fenced JSON, or reasoning-wrapped text."""
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if "action" in value:
                return value
            candidates.append(value)
    if candidates:
        return candidates[0]
    raise ValueError("no JSON object found")


def _enforce_evidence_protocol(
    action: Action,
    state: dict[str, Any],
    observation: ToolObservation | None,
) -> Action:
    """Reject model actions that violate the public evidence policy.

    The model still chooses every semantic action. This guard only converts two
    invalid transitions after a successful search: search hits must be read as
    original evidence, while an empty result must not become an unsupported
    clarification request.
    """
    if not observation or not observation.ok or observation.tool not in {"search_documents", "hybrid_search"}:
        return action

    hits = state.get("hits") or []
    readings = state.get("readings") or []
    if hits and not readings and action.name in {"ask", "finish"}:
        first_doc_id = str(hits[0].get("doc_id", "")).strip()
        if first_doc_id:
            return Action(
                "read",
                doc_id=first_doc_id,
                reason="证据协议校验：检索摘要不能作为引用，先读取最高相关候选的原文",
            )

    if not hits and action.name == "ask":
        gaps = action.gaps or [f"知识库中未检索到与“{state.get('question', '当前问题')}”相关的可靠资料"]
        return Action(
            "finish",
            answer="未能在当前知识库中找到足以回答该问题的可靠资料。请补充相关资料或调整问题后重试。",
            gaps=gaps,
            reason="证据协议校验：检索已完成且没有命中，返回资料不足而不是无依据追问",
        )

    if not hits and action.name == "finish" and not action.gaps:
        return Action(
            "finish",
            answer=action.answer
            or "未能在当前知识库中找到足以回答该问题的可靠资料。请补充相关资料或调整问题后重试。",
            gaps=[f"知识库中未检索到与“{state.get('question', '当前问题')}”相关的可靠资料"],
            reason="证据协议校验：无检索命中时明确标记资料缺口",
        )
    return action

