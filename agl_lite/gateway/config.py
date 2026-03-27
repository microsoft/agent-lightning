"""Gateway route configuration — load from YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Wildcard constant — matches any model name.
WILDCARD = "*"


@dataclass(frozen=True)
class RouteConfig:
    """Single route rule: model_in pattern → model_out with parameter adjustments.

    model_in: exact model name or "*" (wildcard, matches any).
    model_out: target model name or "*" (passthrough, keep original model_in).
    """

    model_in: str
    model_out: str
    params_add: dict[str, Any] = field(default_factory=dict)
    params_drop: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GatewayConfig:
    """Gateway configuration. Loaded once at startup, immutable.

    routes: ordered list of RouteConfig. First match wins.
    Empty list = passthrough for all models (no adjustments).
    """

    routes: list[RouteConfig] = field(default_factory=list)


def load_config(path: str) -> GatewayConfig:
    """Load gateway config from a YAML file.

    Expected format (list-based, priority order):
        routes:
          - model_in: gpt-4          # exact match
            model_out: qwen-7b
            params:
              add:
                temperature: 0.7
                max_tokens: 4096
              drop:
                - frequency_penalty

          - model_in: claude-3       # exact match
            model_out: qwen-7b
            params:
              add:
                temperature: 0.8

          - model_in: "*"            # wildcard catch-all (lowest priority)
            model_out: qwen-7b       # redirect all unmatched to qwen-7b
            params:
              drop:
                - stream_options

    Wildcard rules:
      - model_in: "*" matches any model name not matched by earlier rules.
      - model_out: "*" means keep the original model name (passthrough).
      - Order matters: first match wins. Put specific rules before wildcards.

    Environment variable substitution:
      - ``${VAR_NAME}`` in any value is replaced with the env var's value.
      - Example: ``model_out: "${AGL_MODEL_NAME}"`` resolves at load time.
    """
    raw_text = Path(path).read_text()
    raw_text = os.path.expandvars(raw_text)
    raw = yaml.safe_load(raw_text)
    if not raw or "routes" not in raw:
        return GatewayConfig()

    routes: list[RouteConfig] = []
    for route_raw in raw["routes"]:
        params = route_raw.get("params", {})
        routes.append(
            RouteConfig(
                model_in=str(route_raw["model_in"]),
                model_out=str(route_raw["model_out"]),
                params_add=params.get("add", {}),
                params_drop=params.get("drop", []),
            )
        )
    return GatewayConfig(routes=routes)
