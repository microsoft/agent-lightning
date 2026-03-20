"""Gateway router — model routing, server selection, parameter adjustment."""

from __future__ import annotations

from typing import Any

from agl_lite.gateway.config import GatewayConfig, RouteConfig
from agl_lite.schemas.model_server import ModelServer
from agl_lite.store.memory import InMemoryStore


class NoServersError(Exception):
    """Raised when no servers are available for a model."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"No servers available for model '{model}'")


class GatewayRouter:
    """Routes agent requests to model servers.

    Responsibilities:
      - Resolve model_in → model_out via config (exact match, passthrough if missing)
      - Select a server from the model pool (round-robin)
      - Adjust request body parameters (add/drop fields, rewrite model)

    Thread safety: single-threaded (asyncio event loop), no locks needed.
    """

    def __init__(self, config: GatewayConfig, store: InMemoryStore) -> None:
        self._config = config
        self._store = store
        # Round-robin index per model. TODO: configurable strategy (least-connections, random, etc.)
        self._rr_index: dict[str, int] = {}

    def resolve(self, model_in: str) -> tuple[str, RouteConfig | None]:
        """Resolve model_in to (model_out, route_config).

        Exact match in config routes. If no match, passthrough: model_out = model_in, no adjustments.
        """
        route = self._config.routes.get(model_in)
        if route is not None:
            return route.model_out, route
        return model_in, None

    def select_server(self, model: str) -> ModelServer:
        """Pick a server from the model pool using round-robin.

        Raises NoServersError if pool is empty or model not registered.
        """
        pool = self._store.get_model_pool(model)
        if not pool:
            raise NoServersError(model)

        idx = self._rr_index.get(model, 0) % len(pool)
        self._rr_index[model] = idx + 1
        return pool[idx]

    def prepare_body(self, body: dict[str, Any], model_out: str, route: RouteConfig | None) -> dict[str, Any]:
        """Rewrite model field and apply parameter adjustments.

        - Always sets model to model_out.
        - If route has params.add, merges them (override existing).
        - If route has params.drop, removes those keys.
        """
        body = {**body, "model": model_out}

        if route is not None:
            # Add/override parameters.
            if route.params_add:
                body.update(route.params_add)
            # Drop parameters.
            for key in route.params_drop:
                body.pop(key, None)

        return body
