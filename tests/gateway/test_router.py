"""Tests for gateway router — routing, server selection, param adjustment."""

from __future__ import annotations

import pytest

from agl_lite.gateway.config import GatewayConfig, RouteConfig
from agl_lite.gateway.router import GatewayRouter, NoServersError
from agl_lite.schemas.api import RegisterModelRequest
from agl_lite.store.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def _register(store: InMemoryStore, model: str, endpoint: str, version: int = 0):
    store.register_models([RegisterModelRequest(model=model, endpoint=endpoint, version=version)])


class TestResolve:
    def test_exact_match(self, store: InMemoryStore):
        config = GatewayConfig(routes=[RouteConfig(model_in="gpt-4", model_out="qwen-7b")])
        router = GatewayRouter(config, store)
        model_out, route = router.resolve("gpt-4")
        assert model_out == "qwen-7b"
        assert route is not None

    def test_passthrough_no_routes(self, store: InMemoryStore):
        config = GatewayConfig()
        router = GatewayRouter(config, store)
        model_out, route = router.resolve("qwen-7b")
        assert model_out == "qwen-7b"
        assert route is None

    def test_no_match(self, store: InMemoryStore):
        config = GatewayConfig(routes=[RouteConfig(model_in="gpt-4", model_out="qwen-7b")])
        router = GatewayRouter(config, store)
        model_out, route = router.resolve("claude-3")
        assert model_out == "claude-3"
        assert route is None

    def test_wildcard_redirect(self, store: InMemoryStore):
        """* → model_out: redirect any unmatched model to a specific model."""
        config = GatewayConfig(routes=[RouteConfig(model_in="*", model_out="qwen-7b")])
        router = GatewayRouter(config, store)
        model_out, route = router.resolve("anything")
        assert model_out == "qwen-7b"
        assert route is not None

    def test_wildcard_passthrough(self, store: InMemoryStore):
        """* → *: keep original model name but apply param adjustments."""
        config = GatewayConfig(
            routes=[RouteConfig(model_in="*", model_out="*", params_drop=["stream_options"])]
        )
        router = GatewayRouter(config, store)
        model_out, route = router.resolve("my-model")
        assert model_out == "my-model"
        assert route is not None
        assert route.params_drop == ["stream_options"]

    def test_exact_before_wildcard(self, store: InMemoryStore):
        """Exact match takes priority over wildcard (order matters)."""
        config = GatewayConfig(
            routes=[
                RouteConfig(model_in="gpt-4", model_out="qwen-7b"),
                RouteConfig(model_in="*", model_out="llama-8b"),
            ]
        )
        router = GatewayRouter(config, store)

        # gpt-4 matches exact rule
        model_out, route = router.resolve("gpt-4")
        assert model_out == "qwen-7b"

        # anything else matches wildcard
        model_out, route = router.resolve("claude-3")
        assert model_out == "llama-8b"

    def test_first_match_wins(self, store: InMemoryStore):
        """First matching rule wins, even if later rules also match."""
        config = GatewayConfig(
            routes=[
                RouteConfig(model_in="*", model_out="first"),
                RouteConfig(model_in="*", model_out="second"),
            ]
        )
        router = GatewayRouter(config, store)
        model_out, _ = router.resolve("anything")
        assert model_out == "first"

    def test_wildcard_preserves_model_in(self, store: InMemoryStore):
        """Wildcard passthrough preserves the original model name."""
        config = GatewayConfig(routes=[RouteConfig(model_in="*", model_out="*")])
        router = GatewayRouter(config, store)

        model_out, _ = router.resolve("gpt-4")
        assert model_out == "gpt-4"

        model_out, _ = router.resolve("claude-3")
        assert model_out == "claude-3"


class TestSelectServer:
    def test_single_server(self, store: InMemoryStore):
        _register(store, "qwen-7b", "http://vllm-0:8000/v1")
        router = GatewayRouter(GatewayConfig(), store)
        server = router.select_server("qwen-7b")
        assert server.endpoint == "http://vllm-0:8000/v1"

    def test_round_robin(self, store: InMemoryStore):
        _register(store, "qwen-7b", "http://vllm-0:8000/v1")
        _register(store, "qwen-7b", "http://vllm-1:8000/v1")
        router = GatewayRouter(GatewayConfig(), store)

        endpoints = [router.select_server("qwen-7b").endpoint for _ in range(4)]
        assert endpoints[0] != endpoints[1]
        assert endpoints[0] == endpoints[2]
        assert endpoints[1] == endpoints[3]

    def test_no_servers(self, store: InMemoryStore):
        router = GatewayRouter(GatewayConfig(), store)
        with pytest.raises(NoServersError, match="qwen-7b"):
            router.select_server("qwen-7b")

    def test_round_robin_wraps(self, store: InMemoryStore):
        _register(store, "qwen-7b", "http://vllm-0:8000/v1")
        _register(store, "qwen-7b", "http://vllm-1:8000/v1")
        _register(store, "qwen-7b", "http://vllm-2:8000/v1")
        router = GatewayRouter(GatewayConfig(), store)

        endpoints = [router.select_server("qwen-7b").endpoint for _ in range(6)]
        assert endpoints[0] == endpoints[3]
        assert endpoints[1] == endpoints[4]
        assert endpoints[2] == endpoints[5]

    def test_independent_per_model(self, store: InMemoryStore):
        _register(store, "qwen-7b", "http://qwen-0:8000/v1")
        _register(store, "reward", "http://reward-0:8000/v1")
        router = GatewayRouter(GatewayConfig(), store)

        s1 = router.select_server("qwen-7b")
        s2 = router.select_server("reward")
        assert s1.model == "qwen-7b"
        assert s2.model == "reward"


class TestPrepareBody:
    def test_rewrite_model(self, store: InMemoryStore):
        router = GatewayRouter(GatewayConfig(), store)
        body = router.prepare_body({"model": "gpt-4", "messages": []}, "qwen-7b", None)
        assert body["model"] == "qwen-7b"
        assert body["messages"] == []

    def test_add_params(self, store: InMemoryStore):
        route = RouteConfig(model_in="gpt-4", model_out="qwen-7b", params_add={"temperature": 0.7, "max_tokens": 4096})
        router = GatewayRouter(GatewayConfig(), store)
        body = router.prepare_body({"model": "gpt-4"}, "qwen-7b", route)
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 4096

    def test_drop_params(self, store: InMemoryStore):
        route = RouteConfig(model_in="gpt-4", model_out="qwen-7b", params_drop=["frequency_penalty", "presence_penalty"])
        router = GatewayRouter(GatewayConfig(), store)
        body = router.prepare_body(
            {"model": "gpt-4", "frequency_penalty": 0.5, "presence_penalty": 0.2, "messages": []},
            "qwen-7b",
            route,
        )
        assert "frequency_penalty" not in body
        assert "presence_penalty" not in body
        assert body["messages"] == []

    def test_add_overrides_existing(self, store: InMemoryStore):
        route = RouteConfig(model_in="gpt-4", model_out="qwen-7b", params_add={"temperature": 0.7})
        router = GatewayRouter(GatewayConfig(), store)
        body = router.prepare_body({"model": "gpt-4", "temperature": 0.9}, "qwen-7b", route)
        assert body["temperature"] == 0.7

    def test_drop_missing_key_ok(self, store: InMemoryStore):
        route = RouteConfig(model_in="gpt-4", model_out="qwen-7b", params_drop=["nonexistent"])
        router = GatewayRouter(GatewayConfig(), store)
        body = router.prepare_body({"model": "gpt-4"}, "qwen-7b", route)
        assert body == {"model": "qwen-7b"}

    def test_passthrough_no_adjustment(self, store: InMemoryStore):
        router = GatewayRouter(GatewayConfig(), store)
        original = {"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]}
        body = router.prepare_body(original, "qwen-7b", None)
        assert body == {"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]}

    def test_does_not_mutate_original(self, store: InMemoryStore):
        route = RouteConfig(model_in="gpt-4", model_out="qwen-7b", params_add={"temperature": 0.7}, params_drop=["top_p"])
        router = GatewayRouter(GatewayConfig(), store)
        original = {"model": "gpt-4", "top_p": 0.9, "messages": []}
        router.prepare_body(original, "qwen-7b", route)
        assert original == {"model": "gpt-4", "top_p": 0.9, "messages": []}
