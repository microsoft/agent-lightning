# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from agentlightning.algorithm.base import Algorithm
from agentlightning.algorithm.utils import with_llm_proxy, with_store
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore


class _BaseAlgorithm(Algorithm):
    def run(self, *args: Any, **kwargs: Any) -> None:
        """Satisfy the abstract interface without invoking training logic."""
        return None


class _StubLLMProxy:
    """Test double that tracks lifecycle calls."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.running = False

    def is_running(self) -> bool:
        return self.running

    async def start(self) -> None:
        self.start_calls += 1
        self.running = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


@pytest.mark.asyncio
async def test_with_store_injects_store_argument():
    class StoreAlgorithm(_BaseAlgorithm):
        @with_store
        async def record_store(self, store: LightningStore, payload: str) -> None:
            self.seen_store = store  # type: ignore[attr-defined]
            self.seen_payload = payload  # type: ignore[attr-defined]

    algorithm = StoreAlgorithm()
    fake_store = MagicMock(spec=LightningStore)
    algorithm.set_store(fake_store)

    await algorithm.record_store("batch-1")

    assert algorithm.seen_store is fake_store  # type: ignore[attr-defined]
    assert algorithm.seen_payload == "batch-1"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_with_llm_proxy_allows_optional_injection():
    class OptionalProxyAlgorithm(_BaseAlgorithm):
        @with_llm_proxy()
        async def record_proxy(self, llm_proxy: LLMProxy | None, marker: str) -> None:
            self.seen_proxy = llm_proxy  # type: ignore[attr-defined]
            self.marker = marker  # type: ignore[attr-defined]

    algorithm = OptionalProxyAlgorithm()
    algorithm.set_llm_proxy(None)

    await algorithm.record_proxy("optional")

    assert algorithm.seen_proxy is None  # type: ignore[attr-defined]
    assert algorithm.marker == "optional"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_with_llm_proxy_required_raises_when_missing():
    class RequiredProxyAlgorithm(_BaseAlgorithm):
        @with_llm_proxy(required=True)
        async def record_proxy(self, llm_proxy: LLMProxy) -> None:
            self.seen_proxy = llm_proxy  # type: ignore[attr-defined]

    algorithm = RequiredProxyAlgorithm()
    algorithm.set_llm_proxy(None)

    with pytest.raises(ValueError):
        await algorithm.record_proxy()


@pytest.mark.asyncio
async def test_with_llm_proxy_auto_start_and_stop():
    class AutoProxyAlgorithm(_BaseAlgorithm):
        @with_llm_proxy()
        async def use_proxy(self, llm_proxy: LLMProxy | None) -> None:
            if llm_proxy is None:
                raise AssertionError("LLM proxy should be injected")
            self.seen_proxy = llm_proxy  # type: ignore[attr-defined]

    algorithm = AutoProxyAlgorithm()
    proxy = _StubLLMProxy()
    algorithm.set_llm_proxy(cast(LLMProxy, proxy))

    await algorithm.use_proxy()

    assert algorithm.seen_proxy is proxy  # type: ignore[attr-defined]
    assert proxy.start_calls == 1
    assert proxy.stop_calls == 1

    # When already running, no extra start/stop should be requested.
    proxy.running = True
    await algorithm.use_proxy()

    assert proxy.start_calls == 1
    assert proxy.stop_calls == 1
