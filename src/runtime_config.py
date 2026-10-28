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


class RuntimeModelConfig:
    """Thread-safe, in-memory model configuration with redacted reads."""

    def __init__(self):
        self._lock = threading.RLock()
        self._config = ModelConfig(
            provider=os.getenv("MODEL_PROVIDER", "offline"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

    def public(self) -> dict[str, Any]:
        with self._lock:
            return {
                "provider": self._config.provider,
                "model": self._config.model,
                "base_url": self._config.base_url,
                "api_key_configured": bool(self._config.api_key),
                "persistence": "memory_only",
            }

    def update(self, provider: str, model: str, base_url: str, api_key: str | None = None) -> dict[str, Any]:
        provider = provider.strip().lower()
        model = model.strip()
        base_url = base_url.strip().rstrip("/")
        if provider not in {"offline", "openai", "deepseek"}:
            raise ModelConfigError("provider 仅支持 offline、openai 或 deepseek")
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
            self._config = ModelConfig(provider, model or self._config.model, base_url or self._config.base_url, next_key)
            return self.public()

    def test_connection(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: int = 20,
    ) -> dict[str, Any]:
        provider = provider.strip().lower()
        if provider == "offline":
            return {"ok": True, "provider": "offline", "model": "offline", "latency_ms": 0, "message": "离线决策模型可用，无需网络连接"}
        with self._lock:
            fallback_key = self._config.api_key
        key = api_key.strip() if api_key and api_key.strip() else fallback_key
        # Reuse validation without changing the active configuration.
        if provider not in {"openai", "deepseek"}:
            raise ModelConfigError("仅支持 OpenAI-compatible 或 DeepSeek 连通性测试")
        model = model.strip()
        base_url = base_url.strip().rstrip("/")
        parsed = urlparse(base_url)
        if not model:
            raise ModelConfigError("必须填写模型名称")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ModelConfigError("Base URL 必须是有效的 HTTP(S) 地址")
        if not key:
            raise ModelConfigError("测试在线模型必须填写 API Key")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a connectivity test. Reply with exactly OK."},
                {"role": "user", "content": "ping"},
            ],
            "temperature": 0,
            "max_tokens": 16,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=max(3, min(timeout_seconds, 60))) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
                content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
                return {
                    "ok": True,
                    "provider": provider,
                    "model": data.get("model", model),
                    "base_url": base_url,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "http_status": response.status,
                    "response_preview": content.strip()[:120] or "请求成功（响应正文为空）",
                }
        except urllib.error.HTTPError as exc:
            message = self._safe_error_message(exc)
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url, "latency_ms": round((time.perf_counter() - started) * 1000), "http_status": exc.code, "message": message}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url, "latency_ms": round((time.perf_counter() - started) * 1000), "http_status": None, "message": f"连接失败：{getattr(exc, 'reason', str(exc))}"}

    def _safe_error_message(self, error: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(error.read().decode("utf-8"))
            detail = payload.get("error", payload)
            if isinstance(detail, dict):
                return str(detail.get("message") or detail.get("type") or f"HTTP {error.code}")[:300]
            return str(detail)[:300]
        except Exception:
            return f"远端返回 HTTP {error.code}"

    def build_model(self) -> DecisionModel:
        with self._lock:
            config = ModelConfig(**self._config.__dict__)
        if config.provider in {"openai", "deepseek"}:
            return OpenAICompatibleDecisionModel(config.api_key, config.base_url, config.model)
        return OfflineDecisionModel()

