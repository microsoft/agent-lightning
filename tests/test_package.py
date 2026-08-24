# Copyright (c) Microsoft. All rights reserved.

"""Package and application metadata smoke tests."""

from importlib.metadata import metadata, version

from fastapi.testclient import TestClient

from agentlightning import __version__
from agentlightning.server.app import create_app


def test_package_version() -> None:
    assert __version__ == version("agentlightning")


def test_package_metadata() -> None:
    package_metadata = metadata("agentlightning")

    assert package_metadata["Author"] == "Agent-lightning Team"
    assert package_metadata["Description-Content-Type"] == "text/markdown"
    assert package_metadata["License-Expression"] == "MIT"
    long_description = package_metadata.json["description"]
    assert isinstance(long_description, str)
    assert long_description.startswith('<p align="center">')
    assert set(package_metadata.get_all("Project-URL") or []) == {
        "Documentation, https://microsoft.github.io/agent-lightning/stable/",
        "Homepage, https://github.com/microsoft/agent-lightning",
        "Issues, https://github.com/microsoft/agent-lightning/issues",
        "Repository, https://github.com/microsoft/agent-lightning",
    }


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
