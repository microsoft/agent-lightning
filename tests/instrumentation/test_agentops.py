# Copyright (c) Microsoft. All rights reserved.

# FIXME: This file will have side-effects on other tests if the tests failed and agentops service is not disabled.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.sdk.trace.export import SpanExportResult

import agentlightning.instrumentation.agentops as agentops_instrumentation
from agentlightning.instrumentation.agentops import (
    BypassableAuthenticatedOTLPExporter,
    BypassableOTLPMetricExporter,
    BypassableOTLPSpanExporter,
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
def test_switchable_otlp_span_exporter():

    switchable_otlp_span_exporter = BypassableOTLPSpanExporter()
    with patch.object(
        # BypassableOTLPSpanExporter is a subclass of LightningStoreOTLPExporter, which is a subclass of OTLPSpanExporter
        switchable_otlp_span_exporter.__class__.__bases__[-1].__bases__[0],
        "export",
        return_value=SpanExportResult.SUCCESS,
    ) as mock_export:
        enable_agentops_service()
        result = switchable_otlp_span_exporter.export([])
        assert result == SpanExportResult.SUCCESS
        mock_export.assert_called_once()

        enable_agentops_service(False)
        result = switchable_otlp_span_exporter.export([])
        assert result == SpanExportResult.SUCCESS
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

    agentops_instrumentation._wrap_agentops_openai_streaming(  # pyright: ignore[reportPrivateUsage]
        SimpleNamespace(_tracer=MagicMock())
    )

    assert wrapped == [
        ("openai.resources.chat.completions", "Completions.create"),
        ("openai.resources.chat.completions", "AsyncCompletions.create"),
        ("openai.resources.responses", "Responses.create"),
        ("openai.resources.responses", "AsyncResponses.create"),
    ]
