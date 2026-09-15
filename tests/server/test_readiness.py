# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from fastapi.testclient import TestClient


def test_publish_and_get_k8s_image_readiness(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    monkeypatch.setattr("agentlightning.server.routes.readiness.time.time", lambda: 100.0)

    published = client.put(
        "/api/runner-readiness/k8s",
        headers=auth_headers,
        json={
            "images": ["swebench/repo:openai", "docker.io/swebench/repo:openai", "alpine"],
            "node_count": 2,
            "lease_seconds": 30,
        },
    )

    assert published.status_code == 200
    assert published.json() == {
        "images": ["docker.io/library/alpine:latest", "docker.io/swebench/repo:openai"],
        "node_count": 2,
        "observed_at": 100.0,
        "expires_at": 130.0,
    }
    fetched = client.get("/api/runner-readiness/k8s", headers=auth_headers)
    assert fetched.status_code == 200
    assert fetched.json() == published.json()


def test_get_readiness_returns_503_when_missing(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/api/runner-readiness/k8s", headers=auth_headers)

    assert response.status_code == 503
    assert "not been published" in response.json()["detail"]


def test_get_readiness_returns_503_when_expired(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    ticks = iter([100.0, 131.0])
    monkeypatch.setattr("agentlightning.server.routes.readiness.time.time", lambda: next(ticks))
    published = client.put(
        "/api/runner-readiness/k8s",
        headers=auth_headers,
        json={"images": [], "node_count": 1, "lease_seconds": 30},
    )
    assert published.status_code == 200

    response = client.get("/api/runner-readiness/k8s", headers=auth_headers)

    assert response.status_code == 503
    assert "expired" in response.json()["detail"]


def test_readiness_endpoints_require_auth(client: TestClient) -> None:
    assert client.get("/api/runner-readiness/k8s").status_code == 401
    assert (
        client.put(
            "/api/runner-readiness/k8s",
            json={"images": [], "node_count": 1, "lease_seconds": 30},
        ).status_code
        == 401
    )


def test_publish_rejects_invalid_node_count_and_lease(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.put(
        "/api/runner-readiness/k8s",
        headers=auth_headers,
        json={"images": [], "node_count": 0, "lease_seconds": 301},
    )

    assert response.status_code == 422
