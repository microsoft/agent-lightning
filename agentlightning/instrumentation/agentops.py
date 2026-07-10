# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
import logging
from collections.abc import Callable as CallableABC
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable, cast

import requests
from agentops.client.api import V3Client, V4Client
from agentops.client.api.types import AuthTokenResponse
from agentops.sdk.exporters import AuthenticatedOTLPExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.trace import Tracer

from agentlightning.utils.otlp import LightningStoreOTLPExporter

logger = logging.getLogger(__name__)

__all__ = [
    "instrument_agentops",
    "uninstrument_agentops",
]

# Module-level storage for originals
_original_handle_chat_attributes: Callable[..., Any] | None = None
_original_openai_custom_wrap: Callable[..., Any] | None = None
_agentops_service_enabled = False

_Wrapper = Callable[..., Any]
_WrapperFactory = Callable[[Tracer], _Wrapper]
_wrapt = cast(Any, import_module("wrapt"))
wrap_function_wrapper = cast(Callable[[str, str, _Wrapper], Any], _wrapt.wrap_function_wrapper)


def _module_available(module_name: str) -> bool:
    try:
        import_module(module_name)
    except ModuleNotFoundError:
        return False
    return True


def _wrap_agentops_openai_streaming(instrumentor: object, **kwargs: object) -> None:
    """Install AgentOps streaming wrappers supported by the active OpenAI SDK."""
    _ = kwargs
    from agentops.instrumentation.providers.openai.utils import is_openai_v1

    stream_wrapper_module = cast(
        Any,
        import_module("agentops.instrumentation.providers.openai.stream_wrapper"),
    )
    chat_completion_stream_wrapper = cast(
        _WrapperFactory,
        stream_wrapper_module.chat_completion_stream_wrapper,
    )
    async_chat_completion_stream_wrapper = cast(
        _WrapperFactory,
        stream_wrapper_module.async_chat_completion_stream_wrapper,
    )
    responses_stream_wrapper = cast(
        _WrapperFactory,
        stream_wrapper_module.responses_stream_wrapper,
    )
    async_responses_stream_wrapper = cast(
        _WrapperFactory,
        stream_wrapper_module.async_responses_stream_wrapper,
    )

    tracer = cast(Tracer | None, getattr(instrumentor, "_tracer", None))
    if not is_openai_v1() or tracer is None:
        return

    wrappers: list[tuple[str, str, _WrapperFactory]] = [
        (
            "openai.resources.chat.completions",
            "Completions.create",
            chat_completion_stream_wrapper,
        ),
        (
            "openai.resources.chat.completions",
            "AsyncCompletions.create",
            async_chat_completion_stream_wrapper,
        ),
        (
            "openai.resources.responses",
            "Responses.create",
            responses_stream_wrapper,
        ),
        (
            "openai.resources.responses",
            "AsyncResponses.create",
            async_responses_stream_wrapper,
        ),
    ]
    beta_module = "openai.resources.beta.chat.completions"
    if _module_available(beta_module):
        wrappers.extend(
            [
                (beta_module, "Completions.parse", chat_completion_stream_wrapper),
                (beta_module, "AsyncCompletions.parse", async_chat_completion_stream_wrapper),
            ]
        )
    else:
        logger.debug("OpenAI beta chat completions are unavailable; skipping AgentOps beta streaming wrappers.")

    for module_name, target, wrapper_factory in wrappers:
        if not _module_available(module_name):
            logger.debug("OpenAI module %s is unavailable; skipping AgentOps wrapper %s.", module_name, target)
            continue
        wrap_function_wrapper(module_name, target, wrapper_factory(tracer))


def _patch_agentops_openai_streaming() -> None:
    """Replace AgentOps 0.4.x's all-or-nothing OpenAI streaming setup."""
    global _original_openai_custom_wrap
    if _original_openai_custom_wrap is not None:
        return
    try:
        instrumentor_module = import_module("agentops.instrumentation.providers.openai.instrumentor")
    except ModuleNotFoundError:
        logger.debug("AgentOps OpenAI instrumentor does not expose the 0.4.x streaming setup.")
        return
    instrumentor_cls = cast(Any, instrumentor_module).OpenaiInstrumentor
    if not hasattr(instrumentor_cls, "_custom_wrap"):
        return
    _original_openai_custom_wrap = instrumentor_cls._custom_wrap
    instrumentor_cls._custom_wrap = _wrap_agentops_openai_streaming


def _unpatch_agentops_openai_streaming() -> None:
    global _original_openai_custom_wrap
    if _original_openai_custom_wrap is None:
        return
    instrumentor_module = import_module("agentops.instrumentation.providers.openai.instrumentor")
    instrumentor_cls = cast(Any, instrumentor_module).OpenaiInstrumentor
    instrumentor_cls._custom_wrap = _original_openai_custom_wrap
    _original_openai_custom_wrap = None


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

    sdk_core = cast(Any, agentops.sdk.core)
    client_api = cast(Any, agentops.client.api)
    sdk_core.AuthenticatedOTLPExporter = BypassableAuthenticatedOTLPExporter
    sdk_core.OTLPMetricExporter = BypassableOTLPMetricExporter
    client_api.V3Client = BypassableV3Client
    client_api.V4Client = BypassableV4Client


def _unpatch_exporters():
    import agentops.client.api
    import agentops.sdk.core

    sdk_core = cast(Any, agentops.sdk.core)
    client_api = cast(Any, agentops.client.api)
    sdk_core.AuthenticatedOTLPExporter = AuthenticatedOTLPExporter
    sdk_core.OTLPMetricExporter = OTLPMetricExporter
    client_api.V3Client = V3Client
    client_api.V4Client = V4Client


def _unwrap_raw_response(response: object) -> object:
    parse = getattr(response, "parse", None)
    if isinstance(parse, CallableABC):
        return cast(Callable[[], object], parse)()
    return response


def _as_list(value: object) -> list[object] | None:
    if isinstance(value, (list, tuple)):
        return list(cast(list[object] | tuple[object, ...], value))
    return None


def _as_first_nested_list(value: object) -> list[object] | None:
    values = _as_list(value)
    if not values:
        return None
    return _as_list(values[0])


def _serialize_model_list(value: object) -> str | None:
    values = _as_list(value)
    if values is None:
        return None
    serialized: list[object] = []
    for item in values:
        model_dump = getattr(item, "model_dump", None)
        if not isinstance(model_dump, CallableABC):
            return None
        serialized.append(cast(Callable[[], object], model_dump)())
    return json.dumps(serialized)


def _patch_agentops_chat_attributes() -> bool:
    import agentops.instrumentation.providers.openai.stream_wrapper
    import agentops.instrumentation.providers.openai.wrappers.chat

    global _original_handle_chat_attributes

    if _original_handle_chat_attributes is not None:
        logger.warning("AgentOps already patched. Skipping.")
        return True

    chat_wrappers = cast(Any, agentops.instrumentation.providers.openai.wrappers.chat)
    _original_handle_chat_attributes = cast(Callable[..., Any], chat_wrappers.handle_chat_attributes)
    handle_chat_attributes = _original_handle_chat_attributes

    def _handle_chat_attributes_with_tokens(
        args: Any = None,
        kwargs: Any = None,
        return_value: object = None,
        **kws: Any,
    ) -> dict[str, Any]:
        attributes = cast(
            dict[str, Any],
            handle_chat_attributes(args=args, kwargs=kwargs, return_value=return_value, **kws),
        )

        # Raw-response clients expose a wrapper whose parsed model contains the token fields.
        return_value = _unwrap_raw_response(return_value)

        prompt_token_ids = _as_list(getattr(return_value, "prompt_token_ids", None))
        if prompt_token_ids is not None:
            attributes["prompt_token_ids"] = prompt_token_ids
        response_token_ids = _as_first_nested_list(getattr(return_value, "response_token_ids", None))
        if response_token_ids is not None:
            attributes["response_token_ids"] = response_token_ids

        # LiteLLM Proxy with vLLM return_token_ids may place response token IDs in choices.
        choices = _as_list(getattr(return_value, "choices", None))
        if choices:
            first_choice = choices[0]
            # Token IDs from "choices[0].token_ids"
            if "response_token_ids" not in attributes:
                token_ids = _as_list(getattr(first_choice, "token_ids", None))
                provider_fields = getattr(first_choice, "provider_specific_fields", None)
                if token_ids is None and isinstance(provider_fields, Mapping):
                    provider_mapping = cast(Mapping[object, object], provider_fields)
                    token_ids = _as_list(provider_mapping.get("token_ids"))
                if token_ids is not None:
                    attributes["response_token_ids"] = token_ids

            # log probability
            # This is temporary. We need a unified convention for classifying and naming logprobs.
            logprobs = getattr(first_choice, "logprobs", None)
            content = _serialize_model_list(getattr(logprobs, "content", None))
            refusal = _serialize_model_list(getattr(logprobs, "refusal", None))
            if content is not None:
                attributes["logprobs.content"] = content
            if refusal is not None:
                attributes["logprobs.refusal"] = refusal

        return attributes

    agentops.instrumentation.providers.openai.wrappers.chat.handle_chat_attributes = _handle_chat_attributes_with_tokens
    agentops.instrumentation.providers.openai.stream_wrapper.handle_chat_attributes = (
        _handle_chat_attributes_with_tokens
    )
    logger.info("Patched AgentOps chat attributes to capture token IDs")
    return True


def _unpatch_agentops_chat_attributes() -> None:
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
        logger.info("Restored AgentOps chat attribute handling")


def instrument_agentops() -> bool:
    """
    Instrument agentops to capture token IDs.
    Requires the supported AgentOps instrumentation API from AgentOps 0.4.21 or newer.
    """
    _patch_exporters()
    _patch_agentops_openai_streaming()

    return _patch_agentops_chat_attributes()


def uninstrument_agentops():
    """Uninstrument agentops to stop capturing token IDs."""
    _unpatch_exporters()
    _unpatch_agentops_openai_streaming()

    _unpatch_agentops_chat_attributes()


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
            return cast(MetricExportResult, cast(Any, super()).export(*args, **kwargs))
        else:
            logger.debug("SwitchableOTLPMetricExporter is switched off, skipping export.")
            return MetricExportResult.SUCCESS


class BypassableV3Client(V3Client):
    """
    V3Client with toggleable authentication calls.
    Returns dummy auth response when `_agentops_service_enabled` is False.
    """

    async def fetch_auth_token(self, api_key: str) -> AuthTokenResponse:
        if _agentops_service_enabled:
            return await super().fetch_auth_token(api_key)
        else:
            logger.debug("SwitchableV3Client is switched off, skipping fetch_auth_token request.")
            return AuthTokenResponse(token="dummy", project_id="dummy")


class BypassableV4Client(V4Client):
    """
    V4Client with toggleable post requests.
    Returns dummy response when `_agentops_service_enabled` is False.
    """

    def post(self, path: str, body: str | bytes, headers: dict[str, str] | None = None) -> requests.Response:
        if _agentops_service_enabled:
            return super().post(path, body, headers)
        else:
            logger.debug("SwitchableV4Client is switched off, skipping post request.")
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            return response
