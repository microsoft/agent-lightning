"""Server configuration — loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    """agl-lite serve settings. All from env vars (prefix-free)."""

    host: str = "0.0.0.0"
    port: int = 8080
    agl_key: str = ""  # AGL_KEY — shared API key; empty = auth disabled
    gateway_config: str = ""  # path to gateway YAML config; empty = no routes (passthrough only)
    hooks: str = ""  # path to Python module with RolloutHooks subclass; empty = no hooks

    model_config = {"env_prefix": ""}
