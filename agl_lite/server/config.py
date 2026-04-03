"""Server configuration — loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    """agl-lite serve settings. All from env vars (prefix-free)."""

    agl_key: str = ""  # AGL_KEY — shared API key; empty = auth disabled
    gateway_config: str | None = None  # path to gateway YAML config
    hooks: str | None = None           # path to Python module with RolloutHooks subclass

    model_config = {"env_prefix": ""}
