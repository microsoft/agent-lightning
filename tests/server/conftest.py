"""Shared fixtures for server endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agl_lite.server.app import create_app
from agl_lite.server.store import _events, _models, _rollouts

AGL_KEY = "test-secret-key"
MODEL_NAME = "test-model"


@pytest.fixture(autouse=True)
def clean_store():
    _rollouts.clear()
    _events.clear()
    _models.clear()
    yield
    _rollouts.clear()
    _events.clear()
    _models.clear()


@pytest.fixture
def server_config() -> dict:
    return {
        "key": AGL_KEY,
        "default_proxy": {
            "model_name": MODEL_NAME,
            "train": {"temperature": 1},
            "val": {"temperature": 0},
        },
    }


@pytest.fixture
def app(server_config: dict[str, str]):
    return create_app(server_config)


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AGL_KEY}"}
