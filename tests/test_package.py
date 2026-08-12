# Copyright (c) Microsoft. All rights reserved.

"""Package rename and application metadata smoke tests."""

from fastapi.testclient import TestClient

from agentlightning.server.app import create_app


def test_server_metadata() -> None:
    app = create_app(
        {
            "key": "",
            "default_proxy": {
                "model_name": "test-model",
                "train": {"temperature": 1},
                "val": {"temperature": 0},
            },
        }
    )
    openapi = TestClient(app).get("/openapi.json").json()
    assert openapi["info"] == {"title": "Agent Lightning", "version": "1.0.0"}
