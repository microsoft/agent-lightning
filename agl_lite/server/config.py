"""Server configuration — pure data carrier, no env reads."""

from __future__ import annotations

from pydantic import BaseModel


class ServerSettings(BaseModel):
    """agl-lite serve settings. Constructed explicitly by cli.py."""

    key: str = ""                      # shared API key; empty = auth disabled
    gateway_config: str | None = None  # path to gateway YAML config
    hooks: str | None = None           # path to Python module with RolloutHooks subclass
    log_dir: str | None = None         # directory for log files (server.log) and default archive
