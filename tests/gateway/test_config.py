"""Tests for gateway config loading."""

from __future__ import annotations

from agl_lite.gateway.config import WILDCARD, GatewayConfig, RouteConfig, load_config


class TestRouteConfig:
    def test_defaults(self):
        r = RouteConfig(model_in="gpt-4", model_out="qwen-7b")
        assert r.params_add == {}
        assert r.params_drop == []

    def test_with_params(self):
        r = RouteConfig(model_in="gpt-4", model_out="qwen-7b", params_add={"temperature": 0.7}, params_drop=["top_p"])
        assert r.params_add == {"temperature": 0.7}
        assert r.params_drop == ["top_p"]

    def test_wildcard_model_in(self):
        r = RouteConfig(model_in="*", model_out="qwen-7b")
        assert r.model_in == WILDCARD

    def test_wildcard_model_out(self):
        r = RouteConfig(model_in="*", model_out="*")
        assert r.model_out == WILDCARD


class TestGatewayConfig:
    def test_empty(self):
        c = GatewayConfig()
        assert c.routes == []

    def test_with_routes(self):
        c = GatewayConfig(routes=[RouteConfig(model_in="gpt-4", model_out="qwen-7b")])
        assert len(c.routes) == 1
        assert c.routes[0].model_in == "gpt-4"


class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        config_file = tmp_path / "gateway.yaml"
        config_file.write_text("""
routes:
  - model_in: gpt-4
    model_out: qwen-7b
    params:
      add:
        temperature: 0.7
        max_tokens: 4096
      drop:
        - frequency_penalty
  - model_in: claude-3
    model_out: qwen-7b
    params:
      add:
        temperature: 0.8
""")
        config = load_config(str(config_file))
        assert len(config.routes) == 2

        gpt4 = config.routes[0]
        assert gpt4.model_in == "gpt-4"
        assert gpt4.model_out == "qwen-7b"
        assert gpt4.params_add == {"temperature": 0.7, "max_tokens": 4096}
        assert gpt4.params_drop == ["frequency_penalty"]

        claude = config.routes[1]
        assert claude.model_in == "claude-3"
        assert claude.model_out == "qwen-7b"
        assert claude.params_add == {"temperature": 0.8}
        assert claude.params_drop == []

    def test_load_empty_yaml(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config(str(config_file))
        assert config.routes == []

    def test_load_no_routes(self, tmp_path):
        config_file = tmp_path / "no_routes.yaml"
        config_file.write_text("other_key: 123\n")
        config = load_config(str(config_file))
        assert config.routes == []

    def test_load_no_params(self, tmp_path):
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("""
routes:
  - model_in: gpt-4
    model_out: qwen-7b
""")
        config = load_config(str(config_file))
        r = config.routes[0]
        assert r.model_in == "gpt-4"
        assert r.model_out == "qwen-7b"
        assert r.params_add == {}
        assert r.params_drop == []

    def test_load_wildcard(self, tmp_path):
        config_file = tmp_path / "wildcard.yaml"
        config_file.write_text("""
routes:
  - model_in: gpt-4
    model_out: qwen-7b
  - model_in: "*"
    model_out: qwen-7b
    params:
      drop:
        - stream_options
""")
        config = load_config(str(config_file))
        assert len(config.routes) == 2
        assert config.routes[0].model_in == "gpt-4"
        assert config.routes[1].model_in == "*"
        assert config.routes[1].model_out == "qwen-7b"
        assert config.routes[1].params_drop == ["stream_options"]

    def test_load_wildcard_passthrough(self, tmp_path):
        config_file = tmp_path / "passthrough.yaml"
        config_file.write_text("""
routes:
  - model_in: "*"
    model_out: "*"
    params:
      drop:
        - stream_options
""")
        config = load_config(str(config_file))
        assert len(config.routes) == 1
        assert config.routes[0].model_in == "*"
        assert config.routes[0].model_out == "*"

    def test_order_preserved(self, tmp_path):
        config_file = tmp_path / "order.yaml"
        config_file.write_text("""
routes:
  - model_in: gpt-4
    model_out: qwen-7b
  - model_in: claude-3
    model_out: llama-8b
  - model_in: "*"
    model_out: qwen-7b
""")
        config = load_config(str(config_file))
        assert [r.model_in for r in config.routes] == ["gpt-4", "claude-3", "*"]
