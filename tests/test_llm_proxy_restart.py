# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for restarting the LiteLLM proxy wrapper."""

from __future__ import annotations

from litellm.litellm_core_utils import logging_worker as litellm_logging_worker

from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.memory import InMemoryLightningStore

from .common.network import get_free_port


def test_restart_resets_litellm_logging_worker() -> None:
    """LLMProxy.start() should recreate LiteLLM's logging worker on each run."""

    store = InMemoryLightningStore()
    proxy = LLMProxy(
        port=get_free_port(),
        model_list=[
            {
                "model_name": "dummy-model",
                # The backend is never invoked; only the proxy lifecycle matters here.
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        store=store,
    )

    try:
        proxy.start()
        first_worker = litellm_logging_worker.GLOBAL_LOGGING_WORKER
        proxy.stop()

        proxy.start()
        second_worker = litellm_logging_worker.GLOBAL_LOGGING_WORKER
    finally:
        proxy.stop()

    assert first_worker is not second_worker, "LiteLLM logging worker should be refreshed after restart"
