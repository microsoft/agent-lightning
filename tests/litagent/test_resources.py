# Copyright (c) Microsoft. All rights reserved.

"""Test ProxyLLM resource behavior."""

import pytest

from agentlightning.types import ProxyLLM


def test_proxy_llm_endpoint_direct_access_emits_warning(caplog: pytest.LogCaptureFixture):
    """Test that accessing endpoint directly on ProxyLLM emits a warning."""
    llm = ProxyLLM(
        endpoint="http://localhost:11434",
        model="gpt-4o-arbitrary",
        sampling_parameters={"temperature": 0.7},
    )

    # Accessing endpoint directly should emit a warning
    _ = llm.endpoint
    assert "Accessing 'endpoint' directly on ProxyLLM is discouraged" in caplog.text


def test_proxy_llm_base_url_no_warning(caplog: pytest.LogCaptureFixture):
    """Test that using get_base_url() method does not emit a warning."""
    llm = ProxyLLM(
        endpoint="http://localhost:11434",
        model="gpt-4o-arbitrary",
        sampling_parameters={"temperature": 0.7},
    )

    # Using base_url should not emit a warning
    url = llm.get_base_url("rollout-123", "attempt-456")
    assert url == "http://localhost:11434/rollout/rollout-123/attempt/attempt-456"
    assert "Accessing 'endpoint' directly on ProxyLLM is discouraged" not in caplog.text


def test_proxy_llm_base_url_returns_endpoint_when_none():
    """Ensure base_url returns the original endpoint when rollout and attempt ids are absent."""
    llm = ProxyLLM(
        endpoint="http://localhost:11434/v1",
        model="gpt-4o-arbitrary",
        sampling_parameters={"temperature": 0.7},
    )

    url = llm.get_base_url(None, None)
    assert url == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    ("rollout_id", "attempt_id"),
    [
        (None, "attempt-456"),
        ("rollout-123", None),
        (123, "attempt-456"),
        ("rollout-123", 456),
    ],
)
def test_proxy_llm_base_url_validates_inputs(rollout_id: object, attempt_id: object) -> None:
    """Ensure base_url enforces consistent, string identifiers when ids are provided."""
    llm = ProxyLLM(
        endpoint="http://localhost:11434",
        model="gpt-4o-arbitrary",
        sampling_parameters={"temperature": 0.7},
    )

    with pytest.raises(ValueError, match="must be strings or all be empty"):
        llm.get_base_url(rollout_id, attempt_id)  # type: ignore
