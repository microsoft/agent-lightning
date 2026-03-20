"""Gateway route configuration — load from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RouteConfig:
    """Mapping for a single model_in → model_out with parameter adjustments."""

    model_out: str
    params_add: dict[str, Any] = field(default_factory=dict)
    params_drop: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GatewayConfig:
    """Gateway configuration. Loaded once at startup, immutable.

    routes: exact-match dict from model_in → RouteConfig.
    Missing model_in = passthrough (model_in used as model_out, no param adjustment).
    """

    routes: dict[str, RouteConfig] = field(default_factory=dict)


def load_config(path: str) -> GatewayConfig:
    """Load gateway config from a YAML file.

    Expected format:
        routes:
          gpt-4:                    # model_in (what agent sends)
            model: qwen-7b          # model_out (lookup key in store)
            params:
              add:
                temperature: 0.7
                max_tokens: 4096
              drop:
                - frequency_penalty
          claude-3:
            model: qwen-7b
            params:
              add:
                temperature: 0.8
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not raw or "routes" not in raw:
        return GatewayConfig()

    routes: dict[str, RouteConfig] = {}
    for model_in, route_raw in raw["routes"].items():
        params = route_raw.get("params", {})
        routes[model_in] = RouteConfig(
            model_out=route_raw["model"],
            params_add=params.get("add", {}),
            params_drop=params.get("drop", []),
        )
    return GatewayConfig(routes=routes)
