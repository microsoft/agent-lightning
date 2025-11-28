# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import MagicMock, patch

from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.sdk.trace.export import SpanExportResult

from agentlightning.instrumentation.agentops import (
    BypassableAuthenticatedOTLPExporter,
    BypassableOTLPMetricExporter,
    BypassableOTLPSpanExporter,
    enable_agentops_service,
)


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
