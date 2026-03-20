"""Shared fixtures for server tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agl_lite.server.app import create_app
from agl_lite.server.config import ServerSettings

AGL_KEY = "test-secret-key"


@pytest.fixture
def settings() -> ServerSettings:
    return ServerSettings(agl_key=AGL_KEY)


@pytest.fixture
def app(settings: ServerSettings):
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AGL_KEY}"}
