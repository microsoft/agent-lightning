"""Tests for gateway config loading."""

from __future__ import annotations

from agl_lite.gateway.config import GatewayConfig, RouteConfig, load_config


class TestRouteConfig:
    def test_defaults(self):
        r = RouteConfig(model_out="qwen-7b")
        assert r.params_add == {}
        assert r.params_drop == []

    def test_with_params(self):
        r = RouteConfig(model_out="qwen-7b", params_add={"temperature": 0.7}, params_drop=["top_p"])
        assert r.params_add == {"temperature": 0.7}
        assert r.params_drop == ["top_p"]


class TestGatewayConfig:
    def test_empty(self):
        c = GatewayConfig()
        assert c.routes == {}

    def test_with_routes(self):
        c = GatewayConfig(routes={"gpt-4": RouteConfig(model_out="qwen-7b")})
        assert "gpt-4" in c.routes


class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        config_file = tmp_path / "gateway.yaml"
        config_file.write_text("""
routes:
  gpt-4:
    model: qwen-7b
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
""")
        config = load_config(str(config_file))
        assert len(config.routes) == 2

        gpt4 = config.routes["gpt-4"]
        assert gpt4.model_out == "qwen-7b"
        assert gpt4.params_add == {"temperature": 0.7, "max_tokens": 4096}
        assert gpt4.params_drop == ["frequency_penalty"]

        claude = config.routes["claude-3"]
        assert claude.model_out == "qwen-7b"
        assert claude.params_add == {"temperature": 0.8}
        assert claude.params_drop == []

    def test_load_empty_yaml(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config(str(config_file))
        assert config.routes == {}

    def test_load_no_routes(self, tmp_path):
        config_file = tmp_path / "no_routes.yaml"
        config_file.write_text("other_key: 123\n")
        config = load_config(str(config_file))
        assert config.routes == {}

    def test_load_no_params(self, tmp_path):
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("""
routes:
  gpt-4:
    model: qwen-7b
""")
        config = load_config(str(config_file))
        r = config.routes["gpt-4"]
        assert r.model_out == "qwen-7b"
        assert r.params_add == {}
        assert r.params_drop == []
