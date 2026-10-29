from __future__ import annotations

import os
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .decision import DecisionModel, OfflineDecisionModel, OpenAICompatibleDecisionModel


class ModelConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    protocol: str = "chat_completions"
    reasoning_effort: str = "none"
    response_format: str = "json_object"


class RuntimeModelConfig:
    """Thread-safe, in-memory model configuration with redacted reads."""

    def __init__(self):
        self._lock = threading.RLock()
        self._config = ModelConfig(
            provider=os.getenv("MODEL_PROVIDER", "offline"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            protocol=os.getenv("MODEL_PROTOCOL", "chat_completions"),
            reasoning_effort=os.getenv("MODEL_REASONING_EFFORT", "none"),
            response_format=os.getenv("MODEL_RESPONSE_FORMAT", "json_object"),
        )

    def public(self) -> dict[str, Any]:
        with self._lock:
            return {
                "provider": self._config.provider,
                "model": self._config.model,
                "base_url": self._config.base_url,
                "protocol": self._config.protocol,
                "reasoning_effort": self._config.reasoning_effort,
                "response_format": self._config.response_format,
                "api_key_configured": bool(self._config.api_key),
                "persistence": "memory_only",
            }

    def update(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        protocol: str = "chat_completions",
        reasoning_effort: str = "none",
        response_format: str = "json_object",
    ) -> dict[str, Any]:
        provider = provider.strip().lower()
        model = model.strip()
        base_url = self._normalize_base_url(base_url)
        if provider not in {"offline", "openai", "deepseek", "custom"}:
            raise ModelConfigError("provider 仅支持 offline、openai、deepseek 或 custom")
        self._validate_options(protocol, reasoning_effort, response_format)
        if provider != "offline":
            if not model:
                raise ModelConfigError("必须填写模型名称")
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ModelConfigError("Base URL 必须是有效的 HTTP(S) 地址")
        with self._lock:
            next_key = self._config.api_key if api_key is None or not api_key.strip() else api_key.strip()
            if provider != "offline" and not next_key:
                raise ModelConfigError("在线模型模式必须配置 API Key")
            self._config = ModelConfig(
                provider,
                model or self._config.model,
                base_url or self._config.base_url,
                next_key,
                protocol,
                reasoning_effort,
                response_format,
            )
            return self.public()

    def test_connection(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: int = 20,
        protocol: str = "chat_completions",
        reasoning_effort: str = "none",
        response_format: str = "json_object",
    ) -> dict[str, Any]:
        provider = provider.strip().lower()
        if provider == "offline":
            return {"ok": True, "provider": "offline", "model": "offline", "latency_ms": 0, "message": "离线决策模型可用，无需网络连接"}
        with self._lock:
            fallback_key = self._config.api_key
        key = api_key.strip() if api_key and api_key.strip() else fallback_key
        # Reuse validation without changing the active configuration.
        if provider not in {"openai", "deepseek", "custom"}:
            raise ModelConfigError("仅支持 OpenAI、DeepSeek 或自定义中转站连通性测试")
        self._validate_options(protocol, reasoning_effort, response_format)
        model = model.strip()
        base_url = self._normalize_base_url(base_url)
        parsed = urlparse(base_url)
        if not model:
            raise ModelConfigError("必须填写模型名称")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ModelConfigError("Base URL 必须是有效的 HTTP(S) 地址")
        if not key:
            raise ModelConfigError("测试在线模型必须填写 API Key")

        payload, endpoint = self._test_payload(model, protocol, reasoning_effort, response_format)
        request = urllib.request.Request(
            f"{base_url}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=max(3, min(timeout_seconds, 60))) as response:
                raw = response.read().decode("utf-8").strip()
                latency_ms = round((time.perf_counter() - started) * 1000)
                if not raw:
                    return {
                        "ok": False,
                        "provider": provider,
                        "model": model,
                        "base_url": base_url,
                        "protocol": protocol,
                        "latency_ms": latency_ms,
                        "http_status": response.status,
                        "message": "接口返回空响应，不能用于问答。请核对模型名称，或切换 Chat Completions / Responses API 后重试",
                    }
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return {
                        "ok": False,
                        "provider": provider,
                        "model": model,
                        "base_url": base_url,
                        "protocol": protocol,
                        "latency_ms": latency_ms,
                        "http_status": response.status,
                        "message": "接口返回的不是 JSON，请检查 Base URL 是否为 OpenAI-compatible API 根路径",
                    }
                content = self._response_content(data, protocol)
                if not content.strip():
                    return {
                        "ok": False,
                        "provider": provider,
                        "model": model,
                        "base_url": base_url,
                        "protocol": protocol,
                        "latency_ms": latency_ms,
                        "http_status": response.status,
                        "message": "接口未返回可用文本，当前模型可能不支持所选协议",
                    }
                return {
                    "ok": True,
                    "provider": provider,
                    "model": data.get("model", model),
                    "base_url": base_url,
                    "protocol": protocol,
                    "latency_ms": latency_ms,
                    "http_status": response.status,
                    "response_preview": content.strip()[:120],
                }
        except urllib.error.HTTPError as exc:
            message = self._safe_error_message(exc)
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url, "latency_ms": round((time.perf_counter() - started) * 1000), "http_status": exc.code, "message": message}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url, "latency_ms": round((time.perf_counter() - started) * 1000), "http_status": None, "message": f"连接失败：{getattr(exc, 'reason', str(exc))}"}

    def list_models(
        self,
        provider: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: int = 20,
    ) -> dict[str, Any]:
        provider = provider.strip().lower()
        if provider == "offline":
            return {"ok": True, "provider": "offline", "models": ["offline"], "count": 1, "latency_ms": 0}
        if provider not in {"openai", "deepseek", "custom"}:
            raise ModelConfigError("仅支持 OpenAI、DeepSeek 或自定义中转站模型发现")
        base_url = self._normalize_base_url(base_url)
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ModelConfigError("Base URL 必须是有效的 HTTP(S) 地址")
        with self._lock:
            fallback_key = self._config.api_key
        key = api_key.strip() if api_key and api_key.strip() else fallback_key
        if not key:
            raise ModelConfigError("获取模型列表必须填写 API Key")

        request = urllib.request.Request(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            method="GET",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=max(3, min(timeout_seconds, 60))) as response:
                payload = json.loads(response.read().decode("utf-8"))
                rows = payload.get("data", payload.get("models", [])) if isinstance(payload, dict) else []
                model_ids = []
                for row in rows:
                    value = row.get("id") or row.get("model") or row.get("name") if isinstance(row, dict) else row
                    if value:
                        model_ids.append(str(value))
                models = sorted(set(model_ids), key=str.lower)
                return {
                    "ok": True,
                    "provider": provider,
                    "base_url": base_url,
                    "models": models,
                    "count": len(models),
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "http_status": response.status,
                    "message": "模型列表获取成功" if models else "接口请求成功，但没有返回模型 ID",
                }
        except urllib.error.HTTPError as exc:
            return {"ok": False, "provider": provider, "base_url": base_url, "models": [], "count": 0, "latency_ms": round((time.perf_counter() - started) * 1000), "http_status": exc.code, "message": self._safe_error_message(exc)}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"ok": False, "provider": provider, "base_url": base_url, "models": [], "count": 0, "latency_ms": round((time.perf_counter() - started) * 1000), "http_status": None, "message": f"获取模型列表失败：{getattr(exc, 'reason', str(exc))}"}

    def _safe_error_message(self, error: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(error.read().decode("utf-8"))
            detail = payload.get("error", payload)
            if isinstance(detail, dict):
                return str(detail.get("message") or detail.get("type") or f"HTTP {error.code}")[:300]
            return str(detail)[:300]
        except Exception:
            return f"远端返回 HTTP {error.code}"

    def _normalize_base_url(self, value: str) -> str:
        base_url = value.strip().rstrip("/")
        for suffix in ("/chat/completions", "/responses", "/models"):
            if base_url.endswith(suffix):
                return base_url[:-len(suffix)].rstrip("/")
        return base_url

    def _validate_options(self, protocol: str, reasoning_effort: str, response_format: str) -> None:
        if protocol not in {"chat_completions", "responses"}:
            raise ModelConfigError("接口协议仅支持 Chat Completions 或 Responses API")
        if reasoning_effort not in {"none", "low", "medium", "high", "max"}:
            raise ModelConfigError("推理强度必须是 none/low/medium/high/max")
        if response_format not in {"json_object", "text"}:
            raise ModelConfigError("结构化输出仅支持 json_object 或 text")

    def _test_payload(self, model: str, protocol: str, reasoning_effort: str, response_format: str) -> tuple[dict[str, Any], str]:
        prompt = "Return a JSON object with one field status whose value is OK." if response_format == "json_object" else "Reply with exactly OK."
        if protocol == "responses":
            payload: dict[str, Any] = {"model": model, "input": prompt, "max_output_tokens": 32}
            if reasoning_effort != "none":
                payload["reasoning"] = {"effort": reasoning_effort}
            if response_format == "json_object":
                payload["text"] = {"format": {"type": "json_object"}}
            return payload, "responses"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a connectivity test."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
        }
        if reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload, "chat/completions"

    def _response_content(self, payload: dict[str, Any], protocol: str) -> str:
        if protocol == "chat_completions":
            return payload.get("choices", [{}])[0].get("message", {}).get("content") or ""
        if payload.get("output_text"):
            return str(payload["output_text"])
        texts = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                value = content.get("text") or content.get("output_text")
                if value:
                    texts.append(str(value))
        return "\n".join(texts)

    def build_model(self) -> DecisionModel:
        with self._lock:
            config = ModelConfig(**self._config.__dict__)
        if config.provider in {"openai", "deepseek", "custom"}:
            return OpenAICompatibleDecisionModel(
                config.api_key,
                config.base_url,
                config.model,
                protocol=config.protocol,
                reasoning_effort=config.reasoning_effort,
                response_format=config.response_format,
            )
        return OfflineDecisionModel()

