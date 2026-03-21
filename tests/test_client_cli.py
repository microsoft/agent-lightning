"""Tests for agl-client CLI — run against real FastAPI app via subprocess."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import uvicorn

from agl_lite.server.app import create_app
from agl_lite.server.config import ServerSettings

AGL_KEY = "test-key-cli"


@pytest.fixture(scope="module")
def server():
    """Start agl-lite server in a background thread for CLI testing."""
    import threading
    import time

    settings = ServerSettings(host="127.0.0.1", port=18923, agl_key=AGL_KEY)
    application = create_app(settings)
    config = uvicorn.Config(application, host="127.0.0.1", port=18923, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for server to start.
    time.sleep(1)
    yield
    server.should_exit = True


def _run_cli(*args: str, expect_ok: bool = True) -> str:
    """Run agl-client command and return stdout."""
    env = {
        "AGL_LITE_URL": "http://127.0.0.1:18923",
        "AGL_KEY": AGL_KEY,
        "PATH": "",  # will be overridden
    }
    import os

    full_env = {**os.environ, **env}
    result = subprocess.run(
        [sys.executable, "-m", "agl_lite.client_cli", *args],
        capture_output=True,
        text=True,
        env=full_env,
    )
    if expect_ok:
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return result.stdout


class TestClientCLI:
    def test_health(self, server) -> None:
        out = _run_cli("health")
        assert "OK" in out

    def test_models_list_empty(self, server) -> None:
        out = _run_cli("models", "list")
        assert json.loads(out) == []

    def test_models_register_and_list(self, server) -> None:
        _run_cli("models", "register", "--model", "test-model", "--endpoint", "http://fake:8000/v1")
        out = _run_cli("models", "list")
        models = json.loads(out)
        assert len(models) >= 1
        assert any(m["model"] == "test-model" for m in models)

    def test_models_delete(self, server) -> None:
        _run_cli("models", "register", "--model", "to-delete", "--endpoint", "http://fake:8000/v1")
        _run_cli("models", "delete", "to-delete")
        out = _run_cli("models", "list")
        models = json.loads(out)
        assert not any(m["model"] == "to-delete" for m in models)

    def test_resources_add_and_latest(self, server) -> None:
        _run_cli("resources", "add", '{"job_template": {}, "system_prompt": "test"}')
        out = _run_cli("resources", "latest")
        res = json.loads(out)
        assert "resources_id" in res

    def test_rollouts_list_empty(self, server) -> None:
        out = _run_cli("rollouts", "list")
        rollouts = json.loads(out)
        assert isinstance(rollouts, list)

    def test_events_list_requires_rollout_id(self, server) -> None:
        """Events list without --rollout-id should fail."""
        result = subprocess.run(
            [sys.executable, "-m", "agl_lite.client_cli", "events", "list"],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "AGL_LITE_URL": "http://127.0.0.1:18923", "AGL_KEY": AGL_KEY},
        )
        assert result.returncode != 0
