from __future__ import annotations

import os
import threading
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
        if provider not in {"offline", "openai"}:
            raise ModelConfigError("provider 仅支持 offline 或 openai")
        if provider == "openai":
            if not model:
                raise ModelConfigError("必须填写模型名称")
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ModelConfigError("Base URL 必须是有效的 HTTP(S) 地址")
        with self._lock:
            next_key = self._config.api_key if api_key is None or not api_key.strip() else api_key.strip()
            if provider == "openai" and not next_key:
                raise ModelConfigError("OpenAI-compatible 模式必须配置 API Key")
            self._config = ModelConfig(provider, model or self._config.model, base_url or self._config.base_url, next_key)
            return self.public()

    def build_model(self) -> DecisionModel:
        with self._lock:
            config = ModelConfig(**self._config.__dict__)
        if config.provider == "openai":
            return OpenAICompatibleDecisionModel(config.api_key, config.base_url, config.model)
        return OfflineDecisionModel()

