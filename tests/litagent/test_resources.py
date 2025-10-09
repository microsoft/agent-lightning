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
    """Test that using base_url() method does not emit a warning."""
    llm = ProxyLLM(
        endpoint="http://localhost:11434",
        model="gpt-4o-arbitrary",
        sampling_parameters={"temperature": 0.7},
    )

    # Using base_url should not emit a warning
    url = llm.base_url("rollout-123", "attempt-456")
    assert url == "http://localhost:11434/rollout/rollout-123/attempt/attempt-456"
    assert "Accessing 'endpoint' directly on ProxyLLM is discouraged" not in caplog.text
