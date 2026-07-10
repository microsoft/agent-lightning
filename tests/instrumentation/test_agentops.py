# Copyright (c) Microsoft. All rights reserved.

# FIXME: This file will have side-effects on other tests if the tests failed and agentops service is not disabled.

import json
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.sdk.trace.export import SpanExportResult

import agentlightning.instrumentation.agentops as agentops_instrumentation
from agentlightning.instrumentation.agentops import (
    BypassableAuthenticatedOTLPExporter,
    BypassableOTLPMetricExporter,
    enable_agentops_service,
)


@pytest.mark.agentops
def test_switchable_authenticated_exporter():
    switchable_authenticated_exporter = BypassableAuthenticatedOTLPExporter(endpoint="http://dummy", jwt="dummy")

    with patch.object(
        switchable_authenticated_exporter.__class__.__bases__[-1], "export", return_value=SpanExportResult.SUCCESS
    ) as mock_export:
        enable_agentops_service()
        result = switchable_authenticated_exporter.export([])
        assert result == SpanExportResult.SUCCESS
        mock_export.assert_called_once()

        enable_agentops_service(False)
        result = switchable_authenticated_exporter.export([])
        assert result == SpanExportResult.SUCCESS
        assert mock_export.call_count == 1


@pytest.mark.agentops
def test_switchable_otlp_metric_exporter():

    switchable_otlp_metric_exporter = BypassableOTLPMetricExporter()
    with patch.object(
        switchable_otlp_metric_exporter.__class__.__bases__[-1], "export", return_value=MetricExportResult.SUCCESS
    ) as mock_export:
        enable_agentops_service()
        result = switchable_otlp_metric_exporter.export(metrics_data=MagicMock())
        assert result == MetricExportResult.SUCCESS
        mock_export.assert_called_once()

        enable_agentops_service(False)
        result = switchable_otlp_metric_exporter.export(metrics_data=MagicMock())
        assert result == MetricExportResult.SUCCESS
        assert mock_export.call_count == 1


@pytest.mark.agentops
def test_openai_streaming_wrappers_skip_removed_beta_api(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped: list[tuple[str, str]] = []

    def module_available(module_name: str) -> bool:
        return module_name != "openai.resources.beta.chat.completions"

    def record_wrapper(module_name: str, target: str, wrapper: object) -> None:
        del wrapper
        wrapped.append((module_name, target))

    monkeypatch.setattr(
        agentops_instrumentation,
        "_module_available",
        module_available,
    )
    monkeypatch.setattr(
        agentops_instrumentation,
        "wrap_function_wrapper",
        record_wrapper,
    )

    wrap_streaming = cast(
        Callable[[object], None],
        getattr(agentops_instrumentation, "_wrap_agentops_openai_streaming"),
    )
    wrap_streaming(SimpleNamespace(_tracer=MagicMock()))

    assert wrapped == [
        ("openai.resources.chat.completions", "Completions.create"),
        ("openai.resources.chat.completions", "AsyncCompletions.create"),
        ("openai.resources.responses", "Responses.create"),
        ("openai.resources.responses", "AsyncResponses.create"),
    ]


@pytest.mark.agentops
def test_agentops_chat_patch_extracts_supported_token_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    chat_module = cast(Any, import_module("agentops.instrumentation.providers.openai.wrappers.chat"))
    stream_module = cast(Any, import_module("agentops.instrumentation.providers.openai.stream_wrapper"))

    def base_attributes(**kwargs: object) -> dict[str, Any]:
        del kwargs
        return {}

    monkeypatch.setattr(chat_module, "handle_chat_attributes", base_attributes)
    monkeypatch.setattr(stream_module, "handle_chat_attributes", base_attributes)
    monkeypatch.setattr(agentops_instrumentation, "_original_handle_chat_attributes", None)

    patch_chat = cast(Callable[[], bool], getattr(agentops_instrumentation, "_patch_agentops_chat_attributes"))
    unpatch_chat = cast(Callable[[], None], getattr(agentops_instrumentation, "_unpatch_agentops_chat_attributes"))
    assert patch_chat() is True
    try:
        patched_attributes = cast(Callable[..., dict[str, Any]], chat_module.handle_chat_attributes)
        response = SimpleNamespace(
            prompt_token_ids=[1, 2],
            response_token_ids=[[3, 4]],
            choices=[
                SimpleNamespace(
                    token_ids=None,
                    provider_specific_fields={"token_ids": [5, 6]},
                    logprobs=SimpleNamespace(
                        content=[SimpleNamespace(model_dump=lambda: {"token": "ok"})],
                        refusal=None,
                    ),
                )
            ],
        )

        attributes = patched_attributes(return_value=response)

        assert attributes["prompt_token_ids"] == [1, 2]
        assert attributes["response_token_ids"] == [3, 4]
        assert json.loads(attributes["logprobs.content"]) == [{"token": "ok"}]
    finally:
        unpatch_chat()
