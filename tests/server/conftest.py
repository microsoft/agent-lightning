# Copyright (c) Microsoft. All rights reserved.

"""Shared fixtures for server endpoint tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentlightning.server.app import create_app
from agentlightning.server.store import _events, _models, _rollouts, _runner_readiness, _terminal_order

AGL_KEY = "test-secret-key"
MODEL_NAME = "test-model"


@pytest.fixture(autouse=True)
def clean_store() -> Iterator[None]:
    _rollouts.clear()
    _events.clear()
    _models.clear()
    _terminal_order.clear()
    _runner_readiness.clear()
    yield
    _rollouts.clear()
    _events.clear()
    _models.clear()
    _terminal_order.clear()
    _runner_readiness.clear()


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
def app(server_config: dict[str, Any]) -> FastAPI:
    return create_app(server_config)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AGL_KEY}"}
