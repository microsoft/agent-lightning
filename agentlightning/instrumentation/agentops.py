# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
import logging
from typing import Any, Callable, no_type_check

import requests
from agentops.client.api import V3Client, V4Client
from agentops.client.api.types import AuthTokenResponse
from agentops.sdk.exporters import AuthenticatedOTLPExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics.export import MetricExportResult

from agentlightning.utils.otlp import LightningStoreOTLPExporter

logger = logging.getLogger(__name__)

__all__ = [
    "instrument_agentops",
    "uninstrument_agentops",
]

# Module-level storage for originals
_original_handle_chat_attributes: Callable[..., Any] | None = None
_original_handle_response: Callable[..., Any] | None = None
_agentops_service_enabled = False


def enable_agentops_service(enabled: bool = True) -> None:
    """
    Enable or disable communication with the AgentOps service.

    By default, AgentOps exporters and clients will run in local mode
    and will NOT attempt to communicate with the remote AgentOps service.

    Args:
        enabled: If True, enable all AgentOps exporters and clients.
            All exporters and clients will operate in normal mode and send data
            to the [AgentOps service](https://www.agentops.ai).
    """
    global _agentops_service_enabled
    _agentops_service_enabled = enabled
    logger.info(f"AgentOps service enabled is set to {enabled}.")


def _patch_exporters():
    import agentops.client.api
    import agentops.sdk.core

    agentops.sdk.core.AuthenticatedOTLPExporter = BypassableAuthenticatedOTLPExporter  # type: ignore
    agentops.sdk.core.OTLPMetricExporter = BypassableOTLPMetricExporter
    if hasattr(agentops.sdk.core, "OTLPSpanExporter"):
        agentops.sdk.core.OTLPSpanExporter = BypassableOTLPSpanExporter  # type: ignore
    agentops.client.api.V3Client = BypassableV3Client
    agentops.client.api.V4Client = BypassableV4Client


def _unpatch_exporters():
    import agentops.client.api
    import agentops.sdk.core

    agentops.sdk.core.AuthenticatedOTLPExporter = AuthenticatedOTLPExporter  # type: ignore
    agentops.sdk.core.OTLPMetricExporter = OTLPMetricExporter
    if hasattr(agentops.sdk.core, "OTLPSpanExporter"):
        agentops.sdk.core.OTLPSpanExporter = OTLPSpanExporter  # type: ignore
    agentops.client.api.V3Client = V3Client
    agentops.client.api.V4Client = V4Client


def _unwrap_legacy_response(response: Any) -> Any:
    if hasattr(response, "parse") and callable(response.parse):
        return response.parse()
    return response


def _patch_new_agentops():
    import agentops.instrumentation.providers.openai.stream_wrapper
    import agentops.instrumentation.providers.openai.wrappers.chat
    from agentops.instrumentation.providers.openai.wrappers.chat import handle_chat_attributes  # type: ignore

    global _original_handle_chat_attributes

    if _original_handle_chat_attributes is not None:
        logger.warning("AgentOps already patched. Skipping.")
        return True

    _original_handle_chat_attributes = handle_chat_attributes  # type: ignore

    @no_type_check
    def _handle_chat_attributes_with_tokens(args=None, kwargs=None, return_value=None, **kws):  # type: ignore
        attributes = _original_handle_chat_attributes(args=args, kwargs=kwargs, return_value=return_value, **kws)

        # In some cases, response is a openai._legacy_response.LegacyAPIResponse (e.g., LiteLLM, or LangChain),
        # This is created by client.with_raw_response.create()
        return_value = _unwrap_legacy_response(return_value)

        if (
            return_value is not None
            and hasattr(return_value, "prompt_token_ids")
            and return_value.prompt_token_ids is not None
        ):
            attributes["prompt_token_ids"] = list(return_value.prompt_token_ids)
        if (
            return_value is not None
            and hasattr(return_value, "response_token_ids")
            and return_value.response_token_ids is not None
        ):
            attributes["response_token_ids"] = list(return_value.response_token_ids[0])

        # For LiteLLM Proxy (v0.2) with vLLM return_token_ids, response_token_ids now lives in choices
        if (
            return_value is not None
            and hasattr(return_value, "choices")
            and return_value.choices
            and isinstance(return_value.choices, list)
            and len(return_value.choices) > 0
        ):
            first_choice = return_value.choices[0]
            # Token IDs from "choices[0].token_ids"
            if "response_token_ids" not in attributes:
                if hasattr(first_choice, "token_ids") and first_choice.token_ids is not None:
                    attributes["response_token_ids"] = list(first_choice.token_ids)
                # newer versions of OpenAI client SDK
                elif (
                    hasattr(first_choice, "provider_specific_fields")
                    and first_choice.provider_specific_fields.get("token_ids") is not None
                ):
                    attributes["response_token_ids"] = list(first_choice.provider_specific_fields["token_ids"])

            # log probability
            # This is temporary. We need a unified convention for classifying and naming logprobs.
            if hasattr(first_choice, "logprobs") and first_choice.logprobs is not None:
                if hasattr(first_choice.logprobs, "content") and first_choice.logprobs.content is not None:
                    attributes["logprobs.content"] = json.dumps(
                        [logprob.model_dump() for logprob in first_choice.logprobs.content]
                    )
                if hasattr(first_choice.logprobs, "refusal") and first_choice.logprobs.refusal is not None:
                    attributes["logprobs.refusal"] = json.dumps(
                        [logprob.model_dump() for logprob in first_choice.logprobs.refusal]
                    )

        return attributes

    agentops.instrumentation.providers.openai.wrappers.chat.handle_chat_attributes = _handle_chat_attributes_with_tokens
    agentops.instrumentation.providers.openai.stream_wrapper.handle_chat_attributes = (
        _handle_chat_attributes_with_tokens
    )
    logger.info("Patched newer version of agentops using handle_chat_attributes")
    return True


def _unpatch_new_agentops():
    import agentops.instrumentation.providers.openai.stream_wrapper
    import agentops.instrumentation.providers.openai.wrappers.chat

    global _original_handle_chat_attributes
    if _original_handle_chat_attributes is not None:
        agentops.instrumentation.providers.openai.wrappers.chat.handle_chat_attributes = (
            _original_handle_chat_attributes
        )
        agentops.instrumentation.providers.openai.stream_wrapper.handle_chat_attributes = (
            _original_handle_chat_attributes
        )
        _original_handle_chat_attributes = None
        logger.info("Unpatched newer version of agentops using handle_chat_attributes")


def _patch_old_agentops():
    import opentelemetry.instrumentation.openai.shared.chat_wrappers  # type: ignore
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import _handle_response, dont_throw  # type: ignore

    global _original_handle_response
    _original_handle_response = _handle_response  # type: ignore

    @dont_throw  # type: ignore
    def _handle_response_with_tokens(response, span, *args, **kwargs):  # type: ignore
        _original_handle_response(response, span, *args, **kwargs)  # type: ignore
        if hasattr(response, "prompt_token_ids"):  # type: ignore
            span.set_attribute("prompt_token_ids", list(response.prompt_token_ids))  # type: ignore
        if hasattr(response, "response_token_ids"):  # type: ignore
            span.set_attribute("response_token_ids", list(response.response_token_ids[0]))  # type: ignore

        # For LiteLLM, response is a openai._legacy_response.LegacyAPIResponse
        if hasattr(response, "http_response") and hasattr(response.http_response, "json"):  # type: ignore
            json_data = response.http_response.json()  # type: ignore
            if isinstance(json_data, dict):
                if "prompt_token_ids" in json_data:
                    span.set_attribute("prompt_token_ids", list(json_data["prompt_token_ids"]))  # type: ignore
                if "response_token_ids" in json_data:
                    span.set_attribute("response_token_ids", list(json_data["response_token_ids"][0]))  # type: ignore

    opentelemetry.instrumentation.openai.shared.chat_wrappers._handle_response = _handle_response_with_tokens  # type: ignore
    logger.info("Patched earlier version of agentops using _handle_response")
    return True


def _unpatch_old_agentops():
    import opentelemetry.instrumentation.openai.shared.chat_wrappers  # type: ignore

    global _original_handle_response
    if _original_handle_response is not None:
        opentelemetry.instrumentation.openai.shared.chat_wrappers._handle_response = _original_handle_response  # type: ignore
        _original_handle_response = None
        logger.info("Unpatched earlier version of agentops using _handle_response")


def instrument_agentops():
    """
    Instrument agentops to capture token IDs.
    Automatically detects and uses the appropriate patching method based on the installed agentops version.
    """
    _patch_exporters()

    # Try newest version first (tested for 0.4.16)
    try:
        return _patch_new_agentops()
    except ImportError as e:
        logger.debug(f"Couldn't patch newer version of agentops: {str(e)}")

    # Note: 0.4.15 needs another patching method, but it's too shortlived to be worth handling separately.

    # Try older version (tested for 0.4.13)
    try:
        return _patch_old_agentops()
    except ImportError as e:
        logger.warning(f"Couldn't patch older version of agentops: {str(e)}")
        logger.error("Failed to instrument agentops - neither patching method was successful")
        return False


def uninstrument_agentops():
    """Uninstrument agentops to stop capturing token IDs."""
    _unpatch_exporters()

    try:
        _unpatch_new_agentops()
    except Exception:
        pass
    try:
        _unpatch_old_agentops()
    except Exception:
        pass


class BypassableAuthenticatedOTLPExporter(LightningStoreOTLPExporter, AuthenticatedOTLPExporter):
    """
    AuthenticatedOTLPExporter with switchable service control.

    When `_agentops_service_enabled` is False, skip export and return success.
    """

    def should_bypass(self) -> bool:
        return not _agentops_service_enabled


class BypassableOTLPMetricExporter(OTLPMetricExporter):
    """
    OTLPMetricExporter with switchable service control.
    When `_agentops_service_enabled` is False, skip export and return success.
    """

    def export(self, *args: Any, **kwargs: Any) -> MetricExportResult:
        if _agentops_service_enabled:
            return super().export(*args, **kwargs)  # type: ignore[reportUnknownMemberType]
        else:
            logger.debug("SwitchableOTLPMetricExporter is switched off, skipping export.")
            return MetricExportResult.SUCCESS


class BypassableOTLPSpanExporter(LightningStoreOTLPExporter):
    """
    OTLPSpanExporter with switchable service control.
    When `_agentops_service_enabled` is False, skip export and return success.

    This is used instead of BypassableAuthenticatedOTLPExporter on legacy AgentOps versions.
    """

    def should_bypass(self) -> bool:
        return not _agentops_service_enabled


class BypassableV3Client(V3Client):
    """
    V3Client with toggleable authentication calls.
    Returns dummy auth response when `_agentops_service_enabled` is False.
    """

    # Temporary synchronous override of fetch_auth_token for mock purposes.
    def fetch_auth_token(self, *args: Any, **kwargs: Any) -> AuthTokenResponse:  # type: ignore[override]
        if _agentops_service_enabled:
            return super().fetch_auth_token(*args, **kwargs)  # type: ignore[override]
        else:
            logger.debug("SwitchableV3Client is switched off, skipping fetch_auth_token request.")
            return AuthTokenResponse(token="dummy", project_id="dummy")


class BypassableV4Client(V4Client):
    """
    V4Client with toggleable post requests.
    Returns dummy response when `_agentops_service_enabled` is False.
    """

    def post(self, *args: Any, **kwargs: Any) -> requests.Response:
        if _agentops_service_enabled:
            return super().post(*args, **kwargs)
        else:
            logger.debug("SwitchableV4Client is switched off, skipping post request.")
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            return response
