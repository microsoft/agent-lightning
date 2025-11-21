# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

import litellm
import opentelemetry.trace as trace_api
import yaml
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.proxy.proxy_server import app, save_worker_config  # pyright: ignore[reportUnknownVariableType]
from litellm.types.utils import CallTypes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Scope

from agentlightning.types import LLM, ProxyLLM, SpanNames
from agentlightning.utils.server_launcher import (
    LaunchMode,
    PythonServerLauncher,
    PythonServerLauncherArgs,
    noop_context,
)

from .store.base import LightningStore

logger = logging.getLogger(__name__)

__all__ = [
    "LLMProxy",
]


class ModelConfig(TypedDict):
    """LiteLLM model registration entry.

    This mirrors the items in LiteLLM's `model_list` section.

    Attributes:
        model_name: Logical model name exposed by the proxy.
        litellm_params: Parameters passed to LiteLLM for this model
            (e.g., backend model id, api_base, additional options).
    """  # Google style kept concise.

    model_name: str
    litellm_params: Dict[str, Any]


def _get_pre_call_data(args: Any, kwargs: Any) -> Dict[str, Any]:
    """Extract LiteLLM request payload from hook args.

    The LiteLLM logger hooks receive `(*args, **kwargs)` whose third positional
    argument or `data=` kwarg contains the request payload.

    Args:
        args: Positional arguments from the hook.
        kwargs: Keyword arguments from the hook.

    Returns:
        The request payload dict.

    Raises:
        ValueError: If the payload cannot be located or is not a dict.
    """
    if kwargs.get("data"):
        data = kwargs["data"]
    elif len(args) >= 3:
        data = args[2]
    else:
        raise ValueError(f"Unable to get request data from args or kwargs: {args}, {kwargs}")
    if not isinstance(data, dict):
        raise ValueError(f"Request data is not a dictionary: {data}")
    return cast(Dict[str, Any], data)


def _reset_litellm_logging_worker() -> None:
    """Reset LiteLLM's global logging worker to the current event loop.

    LiteLLM keeps a module-level ``GLOBAL_LOGGING_WORKER`` singleton that owns an
    ``asyncio.Queue``. The queue is bound to the event loop where it was created.
    When the proxy is restarted, Uvicorn spins up a brand new event loop in a new
    thread. If the existing logging worker (and its queue) are reused, LiteLLM
    raises ``RuntimeError: <Queue ...> is bound to a different event loop`` the
    next time it tries to log. Recreating the worker ensures that LiteLLM will
    lazily initialise a fresh queue on the new loop.
    """

    # ``GLOBAL_LOGGING_WORKER`` is imported in a few LiteLLM modules at runtime.
    # Update any already-imported references so future calls use the fresh worker.
    try:
        import litellm.utils as litellm_utils
        from litellm.litellm_core_utils import logging_worker as litellm_logging_worker

        litellm_logging_worker.GLOBAL_LOGGING_WORKER = litellm_logging_worker.LoggingWorker()
        litellm_utils.GLOBAL_LOGGING_WORKER = litellm_logging_worker.GLOBAL_LOGGING_WORKER  # type: ignore[reportAttributeAccessIssue]
    except Exception:  # pragma: no cover - best-effort hygiene
        logger.warning("Unable to propagate LiteLLM logging worker reset.", exc_info=True)


def _reset_litellm_logging_callback_manager() -> None:
    """Reset LiteLLM's global callback manager.

    To get rid of the warning message: "Cannot add callback - would exceed MAX_CALLBACKS limit of 30."
    when litellm is restarted multiple times in the same process.

    It does not respect existing input/output callbacks.
    """

    try:
        litellm.logging_callback_manager._reset_all_callbacks()  # pyright: ignore[reportPrivateUsage]
    except Exception:  # pragma: no cover - best-effort hygiene
        logger.warning("Unable to reset LiteLLM logging callback manager.", exc_info=True)


class AddReturnTokenIds(CustomLogger):
    """LiteLLM logger hook to request token ids from vLLM.

    This mutates the outgoing request payload to include `return_token_ids=True`
    for backends that support token id return (e.g., vLLM).

    See also:
        [vLLM PR #22587](https://github.com/vllm-project/vllm/pull/22587)
    """

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> Optional[Union[Exception, str, Dict[str, Any]]]:
        """Async pre-call hook to adjust request payload.

        Args:
            args: Positional args from LiteLLM.
            kwargs: Keyword args from LiteLLM.

        Returns:
            Either an updated payload dict or an Exception to short-circuit.
        """
        try:
            data = _get_pre_call_data(args, kwargs)
        except Exception as e:
            return e

        # Ensure token ids are requested from the backend when supported.
        return {**data, "return_token_ids": True}


class LightningSpanExporter(SpanExporter):
    """Buffered OTEL span exporter with subtree flushing and training-store sink.

    Design:

    * Spans are buffered until a root span's entire subtree is available.
    * A private event loop on a daemon thread runs async flush logic.
    * Rollout/attempt/sequence metadata is reconstructed by merging headers
      from any span within a subtree.

    Thread-safety:

    * Buffer access is protected by a re-entrant lock.
    * Export is synchronous to the caller yet schedules an async flush on the
      internal loop, then waits for completion.
    """

    def __init__(self, _store: Optional[LightningStore] = None):
        self._store: Optional[LightningStore] = _store  # this is only for testing purposes
        self._buffer: List[ReadableSpan] = []
        self._lock: Optional[threading.Lock] = None
        self._loop_lock_pid: Optional[int] = None

        # Single dedicated event loop running in a daemon thread.
        # This decouples OTEL SDK threads from our async store I/O.
        # Deferred creation until first use.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        self._otlp_exporter = OTLPSpanExporter()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily initialize the event loop and thread on first use.

        Returns:
            asyncio.AbstractEventLoop: The initialized event loop.
        """
        self._clear_loop_and_lock()
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._run_loop, name="LightningSpanExporterLoop", daemon=True)
            self._loop_thread.start()
        return self._loop

    def _ensure_lock(self) -> threading.Lock:
        """Lazily initialize the lock on first use.

        Returns:
            threading.Lock: The initialized lock.
        """
        self._clear_loop_and_lock()
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    def _clear_loop_and_lock(self) -> None:
        """Clear the loop and lock.
        This happens if the exporter was used in a process then used in another process.

        This should only happen in CI.
        """
        if os.getpid() != self._loop_lock_pid:
            logger.warning("Loop and lock are not owned by the current process. Clearing them.")
            self._loop = None
            self._loop_thread = None
            self._lock = None
            self._loop_lock_pid = os.getpid()
        elif self._loop_lock_pid is None:
            self._loop_lock_pid = os.getpid()

    def _run_loop(self) -> None:
        """Run the private asyncio loop forever on the exporter thread."""
        assert self._loop is not None, "Loop should be initialized before thread starts"
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def shutdown(self) -> None:
        """Shut down the exporter event loop.

        Safe to call at process exit.

        """
        if self._loop is None:
            return

        try:

            def _stop():
                assert self._loop is not None
                self._loop.stop()

            self._loop.call_soon_threadsafe(_stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=2.0)
            self._loop.close()
        except Exception:
            logger.exception("Error during exporter shutdown")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans via buffered subtree flush.

        Appends spans to the internal buffer, then triggers an async flush on the
        private event loop. Blocks until that flush completes.

        Args:
            spans: Sequence of spans to export.

        Returns:
            SpanExportResult: SUCCESS on flush success, else FAILURE.
        """
        # Buffer append under lock to protect against concurrent exporters.
        with self._ensure_lock():
            for span in spans:
                self._buffer.append(span)
            default_endpoint = self._otlp_exporter._endpoint  # pyright: ignore[reportPrivateUsage]
            try:
                self._maybe_flush()
            except Exception as e:
                logger.exception("Export flush failed: %s", e)
                return SpanExportResult.FAILURE
            finally:
                self._otlp_exporter._endpoint = default_endpoint  # pyright: ignore[reportPrivateUsage]

        return SpanExportResult.SUCCESS

    def _maybe_flush(self):
        """Flush ready subtrees from the buffer.

        Strategy:
            We consider a subtree "ready" if we can identify a root span. We
            then take that root and all its descendants out of the buffer and
            try to reconstruct rollout/attempt/sequence headers by merging any
            span's `metadata.requester_custom_headers` within the subtree.

        Required headers:
            `x-rollout-id` (str), `x-attempt-id` (str), `x-sequence-id` (str of int)

        Raises:
            None directly. Logs and skips malformed spans.

        """
        # Iterate over current roots. Each iteration pops a whole subtree.
        for root_span_id in self._get_root_span_ids():
            subtree_spans = self._pop_subtrees(root_span_id)
            if not subtree_spans:
                continue

            # Store is initialized lazily here in most cases.
            store = self._store or get_active_llm_proxy().get_store()
            if store is None:
                logger.warning("Store is not set in LLMProxy. Cannot log spans to store.")
                continue

            # If the store supports OTLP endpoint, use it.
            if store.capabilities.get("otlp_traces", False):
                otlp_traces_endpoint = store.otlp_traces_endpoint()
                self._otlp_exporter._endpoint = otlp_traces_endpoint  # pyright: ignore[reportPrivateUsage]
                otlp_enabled = True
            else:
                otlp_enabled = False

            # Merge all custom headers found in the subtree.
            headers_merged: Dict[str, Any] = {}

            for span in subtree_spans:
                if span.attributes is None:
                    continue
                headers_str = span.attributes.get("metadata.requester_custom_headers")
                if headers_str is None:
                    continue
                if not isinstance(headers_str, str):
                    logger.error(
                        f"metadata.requester_custom_headers is not stored as a string: {headers_str}. Skipping the span."
                    )
                    continue
                if not headers_str.strip():
                    logger.warning("metadata.requester_custom_headers is an empty string. Skipping the span.")
                    continue
                try:
                    # Use literal_eval to parse the stringified dict safely.
                    headers = ast.literal_eval(headers_str)
                except Exception as e:
                    logger.error(
                        f"Failed to parse metadata.requester_custom_headers: {headers_str}, error: {e}. Skipping the span."
                    )
                    continue
                if not isinstance(headers, dict):
                    logger.error(
                        f"metadata.requester_custom_headers is not parsed as a dict: {headers}. Skipping the span."
                    )
                    continue
                headers_merged.update(cast(Dict[str, Any], headers))

            if not headers_merged:
                logger.warning(
                    f"No headers found in {len(subtree_spans)} subtree spans of root {root_span_id}. Cannot log to store."
                )
                continue

            # Validate and normalize required header fields.
            rollout_id = headers_merged.get("x-rollout-id")
            attempt_id = headers_merged.get("x-attempt-id")
            sequence_id = headers_merged.get("x-sequence-id")
            if not rollout_id or not attempt_id or not sequence_id or not sequence_id.isdigit():
                logger.warning(
                    f"Missing or invalid rollout_id, attempt_id, or sequence_id in headers: {headers_merged}. Cannot log to store."
                )
                continue
            if not isinstance(rollout_id, str) or not isinstance(attempt_id, str):
                logger.warning(
                    f"rollout_id or attempt_id is not a string: {rollout_id}, {attempt_id}. Cannot log to store."
                )
                continue
            sequence_id_decimal = int(sequence_id)

            # Persist each span in the subtree with the resolved identifiers.
            if otlp_enabled:
                # If store has OTLP support, directly use OTLP exporter and export in batch
                for span in subtree_spans:
                    span._resource = span._resource.merge(  # pyright: ignore[reportPrivateUsage]
                        Resource.create(
                            {
                                SpanNames.ROLLOUT_ID: rollout_id,
                                SpanNames.ATTEMPT_ID: attempt_id,
                                SpanNames.SPAN_SEQUENCE_ID: sequence_id_decimal,
                            }
                        )
                    )
                export_result = self._otlp_exporter.export(subtree_spans)
                if export_result != SpanExportResult.SUCCESS:
                    raise RuntimeError(f"Failed to export spans via OTLP exporter. Result: {export_result}")

            else:
                # The old way: store does not support OTLP endpoint
                for span in subtree_spans:
                    loop = self._ensure_loop()
                    add_otel_span_task = store.add_otel_span(
                        rollout_id=rollout_id,
                        attempt_id=attempt_id,
                        sequence_id=sequence_id_decimal,
                        readable_span=span,
                    )
                    fut = asyncio.run_coroutine_threadsafe(add_otel_span_task, loop)
                    fut.result()  # Bubble up any exceptions from the coroutine.

    def _get_root_span_ids(self) -> Iterable[int]:
        """Yield span_ids for root spans currently in the buffer.

        A root span is defined as one with `parent is None`.

        Yields:
            int: Span id for each root span found.
        """
        for span in self._buffer:
            if span.parent is None:
                span_context = span.get_span_context()
                if span_context is not None:
                    yield span_context.span_id

    def _get_subtrees(self, root_span_id: int) -> Iterable[int]:
        """Yield span_ids in the subtree rooted at `root_span_id`.

        Depth-first traversal over the current buffer.

        Args:
            root_span_id: The span id of the root.

        Yields:
            int: Span ids including the root and all descendants found.
        """
        # Yield the root span id first.
        yield root_span_id
        for span in self._buffer:
            # Check whether the span's parent is the root_span_id.
            if span.parent is not None and span.parent.span_id == root_span_id:
                span_context = span.get_span_context()
                if span_context is not None:
                    # Recursively get child spans.
                    yield from self._get_subtrees(span_context.span_id)

    def _pop_subtrees(self, root_span_id: int) -> List[ReadableSpan]:
        """Remove and return the subtree for a particular root from the buffer.

        Args:
            root_span_id: Root span id identifying the subtree.

        Returns:
            list[ReadableSpan]: Spans that were part of the subtree. Order follows buffer order.
        """
        subtree_span_ids = set(self._get_subtrees(root_span_id))
        subtree_spans: List[ReadableSpan] = []
        new_buffer: List[ReadableSpan] = []
        for span in self._buffer:
            span_context = span.get_span_context()
            if span_context is not None and span_context.span_id in subtree_span_ids:
                subtree_spans.append(span)
            else:
                new_buffer.append(span)
        # Replace buffer with remaining spans to avoid re-processing.
        self._buffer = new_buffer
        return subtree_spans


class LightningOpenTelemetry(OpenTelemetry):
    """OpenTelemetry integration that exports spans to the Lightning store.

    Responsibilities:

    * Ensures each request is annotated with a per-attempt sequence id so spans
      are ordered deterministically even with clock skew across nodes.
    * Uses [`LightningSpanExporter`][agentlightning.llm_proxy.LightningSpanExporter] to persist spans for analytics and training.
    """

    def __init__(self):
        config = OpenTelemetryConfig(exporter=LightningSpanExporter())

        # Check for tracer initialization
        if _check_tracer_provider():
            logger.error("Tracer is already initialized. OpenTelemetry may not work as expected.")

        super().__init__(config=config)  # pyright: ignore[reportUnknownMemberType]

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes] = None
    ) -> Optional[Dict[str, Any]]:
        """The root span is sometimes missing (e.g., when Anthropic endpoint is used).
        It is created in an auth module in LiteLLM. If it's missing, we create it here.
        """
        if "metadata" not in kwargs or "litellm_parent_otel_span" not in kwargs["metadata"]:
            parent_otel_span = self.create_litellm_proxy_request_started_span(  # type: ignore
                start_time=datetime.now(),
                headers=kwargs.get("headers", {}),
            )
            updated_metadata = {**kwargs.get("metadata", {}), "litellm_parent_otel_span": parent_otel_span}

            return {**kwargs, "metadata": updated_metadata}
        else:
            return kwargs


class RolloutAttemptMiddleware(BaseHTTPMiddleware):
    """
    Rewrites /rollout/{rid}/attempt/{aid}/... -> /...
    and injects x-rollout-id, x-attempt-id, x-sequence-id headers.

    LLMProxy can update store later without rebuilding middleware.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Decode rollout and attempt from the URL prefix. Example:
        #   /rollout/r123/attempt/a456/v1/chat/completions
        # becomes
        #   /v1/chat/completions
        # while adding request-scoped headers for trace attribution.
        path = request.url.path

        match = re.match(r"^/rollout/([^/]+)/attempt/([^/]+)(/.*)?$", path)
        if match:
            rollout_id = match.group(1)
            attempt_id = match.group(2)
            new_path = match.group(3) if match.group(3) is not None else "/"

            # Rewrite the ASGI scope path so downstream sees a clean OpenAI path.
            request.scope["path"] = new_path
            request.scope["raw_path"] = new_path.encode()

            store = get_active_llm_proxy().get_store()
            if store is not None:
                # Allocate a monotonic sequence id per (rollout, attempt).
                sequence_id = await store.get_next_span_sequence_id(rollout_id, attempt_id)

                # Inject headers so downstream components and exporters can retrieve them.
                request.scope["headers"] = list(request.scope["headers"]) + [
                    (b"x-rollout-id", rollout_id.encode()),
                    (b"x-attempt-id", attempt_id.encode()),
                    (b"x-sequence-id", str(sequence_id).encode()),
                ]
            else:
                logger.warning("Store is not set. Skipping sequence id allocation and header injection.")

        response = await call_next(request)
        return response


class MessageInspectionMiddleware(BaseHTTPMiddleware):
    """Middleware to inspect the request and response bodies.

    It's for debugging purposes. Add it via "message_inspection" middleware alias.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        ti = time.time()
        logger.info(f"Received request with scope: {request.scope}")
        logger.info(f"Received request with body: {await request.body()}")
        response = await call_next(request)
        elapsed = time.time() - ti
        logger.info(f"Response to request took {elapsed} seconds")
        logger.info(f"Received response with status code: {response.status_code}")
        logger.info(f"Received response with body: {response.body}")
        return response


class StreamConversionMiddleware(BaseHTTPMiddleware):
    """Middleware to convert streaming responses to non-streaming responses.

    Useful for backend that only supports non-streaming responses.

    LiteLLM's OpenTelemetry is also buggy with streaming responses.
    The conversion will hopefully bypass the bug.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Only process POST requests to completion endpoints
        if request.method != "POST":
            return await call_next(request)

        # Check if it's a chat completions or messages endpoint
        endpoint_format: Literal["openai", "anthropic", "unknown"] = "unknown"
        if request.url.path.endswith("/chat/completions") or "/chat/completions?" in request.url.path:
            endpoint_format = "openai"
        elif request.url.path.endswith("/messages") or "/messages?" in request.url.path:
            endpoint_format = "anthropic"
        else:
            endpoint_format = "unknown"

        if endpoint_format == "unknown":
            # Directly bypass the middleware
            return await call_next(request)

        # Read the request body
        try:
            json_body = await request.json()
        except json.JSONDecodeError:
            logger.warning(f"Request body is not valid JSON: {request.body}")
            return await call_next(request)

        # Check if streaming is requested
        is_streaming = json_body.get("stream", False)

        # Simple case: no streaming requested, just return the response
        if not is_streaming:
            return await call_next(request)

        # Now the stream case
        return await self._handle_stream_case(request, json_body, endpoint_format, call_next)

    async def _handle_stream_case(
        self,
        request: Request,
        json_body: Dict[str, Any],
        endpoint_format: Literal["openai", "anthropic"],
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # 1) Modify the request body to force stream=False
        modified_json = dict(json_body)
        modified_json["stream"] = False
        modified_body = json.dumps(modified_json).encode("utf-8")

        # 2) Build a new scope + receive that yields our modified body
        scope: Scope = dict(request.scope)
        # rewrite headers for accept/content-length
        new_headers: List[Tuple[bytes, bytes]] = []
        saw_accept = False
        for k, v in scope["headers"]:
            kl = k.lower()
            if kl == b"accept":
                saw_accept = True
                new_headers.append((k, b"application/json"))
            elif kl == b"content-length":
                # replace with new length
                continue
            else:
                new_headers.append((k, v))
        if not saw_accept:
            new_headers.append((b"accept", b"application/json"))
        new_headers.append((b"content-length", str(len(modified_body)).encode("ascii")))
        scope["headers"] = new_headers

        # Directly modify the request body
        # Creating a new request won't work because request is cached in the base class
        request._body = modified_body  # type: ignore

        response = await call_next(request)

        buffered: Optional[bytes] = None
        # 4) If OK, buffer the response body (it should be JSON because we forced stream=False)
        if 200 <= response.status_code < 300:
            try:
                if hasattr(response, "body_iterator"):
                    # Buffer body safely
                    body_chunks: List[bytes] = []
                    async for chunk in response.body_iterator:  # type: ignore
                        body_chunks.append(chunk)  # type: ignore
                    buffered = b"".join(body_chunks)
                else:
                    buffered = response.body  # type: ignore

                data = json.loads(buffered or b"{}")

                if endpoint_format == "anthropic":
                    return StreamingResponse(
                        self.anthropic_stream_generator(data),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                    )
                else:
                    # openai format
                    return StreamingResponse(
                        self.openai_stream_generator(data),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                    )
            except Exception as e:
                # If anything goes wrong, fall back to non-streaming JSON
                logger.exception(f"Error converting to stream; returning non-stream response: {e}")
                # Rebuild the consumed response
                return Response(
                    content=buffered if buffered is not None else b"",
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                    background=response.background,
                )
        else:
            return response

    async def anthropic_stream_generator(self, original_response: Dict[str, Any]):
        """Generate Anthropic SSE-formatted chunks from complete content blocks

        This is a dirty hack for Anthropic-style streaming from non-streaming response.
        The sse format is subject to change based on Anthropic's implementation.
        If so, try to use `MessageInspectionMiddleware` to inspect the update and fix accordingly.
        """
        # Anthropic format - handle multiple content blocks (text + tool_use)
        content_blocks: List[Dict[str, Any]] = original_response.get("content", [])
        message_id = original_response.get("id", f"msg_{int(time.time() * 1000)}")
        model = original_response.get("model", "claude")

        # Send message_start event
        message_start: Dict[str, Any] = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": original_response.get("usage", {"input_tokens": 0, "output_tokens": 0}),
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Send ping to keep connection alive
        ping = {"type": "ping"}
        yield f"event: ping\ndata: {json.dumps(ping)}\n\n"

        # Process each content block
        for block_index, block in enumerate(content_blocks):
            block_type = block.get("type", "text")

            if block_type == "text":
                # Handle text block
                content = block.get("text", "")

                # Send content_block_start event
                content_block_start = {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

                # Stream text content in chunks
                if content:
                    words = content.split()
                    chunk_size = 5

                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i : i + chunk_size]
                        text_chunk = " ".join(chunk_words)

                        # Add space after chunk unless it's the last one
                        if i + chunk_size < len(words):
                            text_chunk += " "

                        content_block_delta = {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": text_chunk},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n"
                        await asyncio.sleep(0.02)

                # Send content_block_stop event
                content_block_stop = {"type": "content_block_stop", "index": block_index}
                yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"

            elif block_type == "tool_use":
                # Handle tool_use block
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                tool_id = block.get("id", f"toolu_{int(time.time() * 1000)}")

                # Send content_block_start event for tool use
                content_block_start: Dict[str, Any] = {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
                }
                yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

                # Stream tool input as JSON string chunks
                input_json = json.dumps(tool_input)
                chunk_size = 20  # characters per chunk for JSON

                for i in range(0, len(input_json), chunk_size):
                    json_chunk = input_json[i : i + chunk_size]

                    content_block_delta = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": json_chunk},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n"
                    await asyncio.sleep(0.01)

                # Send content_block_stop event
                content_block_stop = {"type": "content_block_stop", "index": block_index}
                yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"

        # Send message_delta event with stop reason
        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": original_response.get("stop_reason", "end_turn"), "stop_sequence": None},
            "usage": {"output_tokens": original_response.get("usage", {}).get("output_tokens", 0)},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

        # Send message_stop event
        message_stop = {"type": "message_stop"}
        yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"

    async def openai_stream_generator(self, response_json: Dict[str, Any]) -> AsyncGenerator[str, Any]:
        """
        Convert a *complete* OpenAI chat.completions choice into a stream of
        OpenAI-compatible SSE chunks.

        This emits:

          - an initial delta with the role ("assistant"),
          - a sequence of deltas for message.content (split into small chunks),
          - deltas for any tool_calls (including id/name and chunked arguments),
          - a terminal chunk with finish_reason,
          - and finally the literal '[DONE]'.

        Notes:

        - We only handle a *single* choice (index 0 typically).
        - We purposefully don't attempt to stream logprobs.
        - Chunking strategy is simple and conservative to avoid splitting
          multi-byte characters: we slice on spaces where possible, then fall
          back to fixed-size substrings.
        """
        choice = cast(Dict[str, Any], (response_json.get("choices") or [{}])[0])
        model = response_json.get("model", "unknown")
        created: int = int(time.time())
        index: int = choice.get("index", 0)

        message: Dict[str, Any] = choice.get("message", {}) or {}
        role: str = message.get("role", "assistant")
        content: str = message.get("content") or ""
        tool_calls: List[Any] = message.get("tool_calls") or []
        finish_reason: Optional[str] = choice.get(
            "finish_reason"
        )  # e.g., "stop", "length", "tool_calls", "content_filter"

        def sse_chunk(obj: Dict[str, Any]) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        # 1) initial chunk with the role
        yield sse_chunk(
            {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": index, "delta": {"role": role}, "finish_reason": None}],
            }
        )

        # 2) stream textual content as small deltas
        async def stream_content(text: str):
            if not text:
                return
            # prefer splitting on spaces in ~20–40 char pieces
            approx = 28
            start = 0
            n = len(text)
            while start < n:
                end = min(start + approx, n)
                if end < n:
                    # try to break on a space going forward
                    space = text.rfind(" ", start, end)
                    if space > start:
                        end = space + 1
                delta_text = text[start:end]
                start = end
                if not delta_text:
                    break
                yield sse_chunk(
                    {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": index, "delta": {"content": delta_text}, "finish_reason": None}],
                    }
                )
                # tiny pause helps some UIs animate smoothly; keep very small
                await asyncio.sleep(0.0)

        async for piece in stream_content(content):  # type: ignore[misc]
            yield piece  # pass through the produced chunks

        # 3) stream tool_calls if present (id/name first, then arguments piecemeal)
        for tc_index, tc in enumerate(tool_calls):
            tc_type = tc.get("type", "function")
            tc_id = tc.get("id") or f"call_{created}_{tc_index}"
            fn: Dict[str, Any] = (tc.get("function") or {}) if tc_type == "function" else {}
            fn_name: str = fn.get("name", "")
            fn_args: str = fn.get("arguments", "") or ""

            # (a) delta that announces the tool call id/type/name
            yield sse_chunk(
                {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": {
                                "tool_calls": [
                                    {"index": tc_index, "id": tc_id, "type": tc_type, "function": {"name": fn_name}}
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

            # (b) stream arguments in small substrings
            arg_chunk_size = 40
            for pos in range(0, len(fn_args), arg_chunk_size):
                partial = fn_args[pos : pos + arg_chunk_size]
                yield sse_chunk(
                    {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": index,
                                "delta": {"tool_calls": [{"index": tc_index, "function": {"arguments": partial}}]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                await asyncio.sleep(0.0)

        # 4) terminal chunk with finish_reason (default to "stop" if missing)
        yield sse_chunk(
            {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": index,
                        "delta": {},
                        "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
                    }
                ],
            }
        )

        # 5) literal DONE sentinel
        yield "data: [DONE]\n\n"


_MIDDLEWARE_REGISTRY: Dict[str, Type[BaseHTTPMiddleware]] = {
    "rollout_attempt": RolloutAttemptMiddleware,
    "stream_conversion": StreamConversionMiddleware,
    "message_inspection": MessageInspectionMiddleware,
}


_CALLBACK_REGISTRY = {
    "return_token_ids": AddReturnTokenIds,
    "opentelemetry": LightningOpenTelemetry,
}


class LLMProxy:
    """Host a LiteLLM OpenAI-compatible proxy bound to a LightningStore.

    The proxy:

    * Serves an OpenAI-compatible API via uvicorn.
    * Adds rollout/attempt routing and headers via middleware.
    * Registers OTEL export and token-id callbacks.
    * Writes a LiteLLM worker config file with `model_list` and settings.

    Lifecycle:

    * [`start()`][agentlightning.LLMProxy.start] writes config, starts uvicorn server in a thread, and waits until ready.
    * [`stop()`][agentlightning.LLMProxy.stop] tears down the server and removes the temp config file.
    * [`restart()`][agentlightning.LLMProxy.restart] convenience wrapper to stop then start.

    !!! note

        As the LLM Proxy sets up an OpenTelemetry tracer, it's recommended to run it in a different
        process from the main runner (i.e., tracer from agents). See `launch_mode` for how to change that.

    !!! warning

        By default (or when "stream_conversion" middleware is enabled), the LLM Proxy will convert OpenAI and Anthropic requests with `stream=True`
        to a non-streaming request before going through the LiteLLM proxy. This is because the OpenTelemetry tracer provided by
        LiteLLM is buggy with streaming responses. You can disable this by removing the "stream_conversion" middleware.
        In that case, you might lose some tracing information like token IDs.

    !!! danger

        Do not run LLM proxy in the same process as the main runner. It's easy to cause conflicts in the tracer provider
        with tracers like [`AgentOpsTracer`][agentlightning.AgentOpsTracer].

    Args:
        port: TCP port to bind. Will bind to a random port if not provided.
        model_list: LiteLLM `model_list` entries.
        store: LightningStore used for span sequence and persistence.
        host: Publicly reachable host used in resource endpoints. See `host` of `launcher_args` for more details.
        litellm_config: Extra LiteLLM proxy config merged with `model_list`.
        num_retries: Default LiteLLM retry count injected into `litellm_settings`.
        num_workers: Number of workers to run in the server. Only applicable for "mp" launch mode. Ignored if launcher_args is provided.
            When `num_workers > 1`, the server will be run using [gunicorn](https://gunicorn.org/).
        launch_mode: Launch mode for the server. Defaults to "mp". Cannot be used together with launcher_args. Ignored if launcher_args is provided.
            It's recommended to use `launch_mode="mp"` to launch the proxy, which will launch the server in a separate process.
            `launch_mode="thread"` can also be used if used in caution. It will launch the server in a separate thread.
            `launch_mode="asyncio"` launches the server in the current thread as an asyncio task.
            It is NOT recommended because it often causes hanging requests. Only use it if you know what you are doing.
        launcher_args: Arguments for the server launcher. If this is provided, host, port, and launch_mode will be ignored. Cannot be used together with port, host, and launch_mode.
        middlewares: List of FastAPI middleware classes or strings to register. You can specify the class aliases or classes that have been imported.
            If not provided, the default middlewares (RolloutAttemptMiddleware and StreamConversionMiddleware) will be used.
            Available middleware aliases are: "rollout_attempt", "stream_conversion", "message_inspection".
            Middlewares are the **first layer** of request processing. They are applied to all requests before the LiteLLM proxy.
        callbacks: List of LiteLLM callback classes or strings to register. You can specify the class aliases or classes that have been imported.
            If not provided, the default callbacks (AddReturnTokenIds and LightningOpenTelemetry) will be used.
            Available callback aliases are: "return_token_ids", "opentelemetry".
    """

    def __init__(
        self,
        port: int | None = None,
        model_list: List[ModelConfig] | None = None,
        store: Optional[LightningStore] = None,
        host: str | None = None,
        litellm_config: Dict[str, Any] | None = None,
        num_retries: int = 0,
        num_workers: int = 1,
        launch_mode: LaunchMode = "mp",
        launcher_args: PythonServerLauncherArgs | None = None,
        middlewares: List[Union[Type[BaseHTTPMiddleware], str]] | None = None,
        callbacks: List[Union[Type[CustomLogger], str]] | None = None,
    ):
        self.store = store

        if launcher_args is not None and (
            port is not None or host is not None or launch_mode != "mp" or num_workers != 1
        ):
            raise ValueError("port, host, launch_mode, and num_workers cannot be set when launcher_args is provided.")

        self.server_launcher_args = launcher_args or PythonServerLauncherArgs(
            port=port,
            host=host,
            launch_mode=launch_mode,
            n_workers=num_workers,
            # NOTE: This /health endpoint can be slow sometimes because it actually probes the backend LLM service.
            healthcheck_url="/health",
            startup_timeout=60.0,
        )

        if self.server_launcher_args.healthcheck_url is None:
            logger.warning("healthcheck_url is not set. LLM Proxy will not be checked for healthiness after starting.")

        self.model_list = model_list or []
        self.litellm_config = litellm_config or {}

        # Ensure num_retries is present inside the litellm_settings block.
        self.litellm_config.setdefault("litellm_settings", {})
        self.litellm_config["litellm_settings"].setdefault("num_retries", num_retries)
        self.server_launcher = PythonServerLauncher(app, self.server_launcher_args, noop_context())

        self._config_file = None

        self.middlewares: List[Type[BaseHTTPMiddleware]] = []
        if middlewares is None:
            middlewares = ["rollout_attempt", "stream_conversion"]
        for middleware in middlewares:
            if isinstance(middleware, str):
                if middleware not in _MIDDLEWARE_REGISTRY:
                    raise ValueError(
                        f"Invalid middleware alias: {middleware}. Available aliases are: {list(_MIDDLEWARE_REGISTRY.keys())}"
                    )
                middleware = _MIDDLEWARE_REGISTRY[middleware]
                self.middlewares.append(middleware)
            else:
                self.middlewares.append(middleware)

        self.callbacks: List[Type[CustomLogger]] = []
        if callbacks is None:
            callbacks = ["return_token_ids", "opentelemetry"]
        for callback in callbacks:
            if isinstance(callback, str):
                if callback not in _CALLBACK_REGISTRY:
                    raise ValueError(
                        f"Invalid callback alias: {callback}. Available aliases are: {list(_CALLBACK_REGISTRY.keys())}"
                    )
                callback = _CALLBACK_REGISTRY[callback]
                self.callbacks.append(callback)
            else:
                self.callbacks.append(callback)

    def get_store(self) -> Optional[LightningStore]:
        """Get the store used by the proxy.

        Returns:
            The store used by the proxy.
        """
        return self.store

    def set_store(self, store: LightningStore) -> None:
        """Set the store for the proxy.

        Args:
            store: The store to use for the proxy.
        """
        self.store = store

    def update_model_list(self, model_list: List[ModelConfig]) -> None:
        """Replace the in-memory model list.

        Args:
            model_list: New list of model entries.
        """
        self.model_list = model_list
        logger.info(f"Updating LLMProxy model list to: {model_list}")
        # Do nothing if the server is not running.

    def initialize(self):
        """Initialize global middleware and LiteLLM callbacks.

        Installs:

        * A FastAPI middleware that rewrites /rollout/{rid}/attempt/{aid}/... paths,
        injects rollout/attempt/sequence headers, and forwards downstream.
        * LiteLLM callbacks for token ids and OpenTelemetry export.

        The middleware can only be installed once because once the FastAPI app has started,
        the middleware cannot be changed any more.

        This function does not start any server. It only wires global hooks.
        """
        if self.store is None:
            raise ValueError("Store is not set. Please set the store before initializing the LLMProxy.")

        if _global_llm_proxy is not None:
            logger.warning("A global LLMProxy is already set. Overwriting it with the new instance.")

        # Set the global LLMProxy reference for middleware/exporter access.
        set_active_llm_proxy(self)

        # Install middleware if it's not already installed.
        installation_status: Dict[Any, bool] = {}
        for mw in app.user_middleware:
            installation_status[mw.cls] = True

        for mw in self.middlewares:
            if mw not in installation_status:
                logger.info(f"Adding middleware {mw} to the FastAPI app.")
                app.add_middleware(mw)
            else:
                logger.info(f"Middleware {mw} is already installed. Will not install a new one.")

        if not initialize_llm_callbacks(self.callbacks):
            # If it's not the first time to initialize the callbacks, also
            # reset LiteLLM's logging worker so its asyncio.Queue binds to the new loop.
            _reset_litellm_logging_worker()

    @asynccontextmanager
    async def _serve_context(self) -> AsyncGenerator[None, None]:
        """Context manager to serve the proxy server.

        See [`start`][agentlightning.LLMProxy.start] and [`stop`][agentlightning.LLMProxy.stop] for more details.
        """

        if not self.store:
            raise ValueError("Store is not set. Please set the store before starting the LLMProxy.")

        # Initialize global middleware and callbacks.
        self.initialize()

        # Persist a temp worker config for LiteLLM and point the proxy at it.
        self._config_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
        with open(self._config_file, "w") as fp:
            yaml.safe_dump(
                {
                    "model_list": self.model_list,
                    **self.litellm_config,
                },
                fp,
            )

        save_worker_config(config=self._config_file)

        # NOTE: When running the _serve_context in current process, you might encounter the following problems:
        # Problem 1: in litellm worker, <Queue at 0x70f1d028cd90 maxsize=50000> is bound to a different event loop
        # Problem 2: Proxy has conflicted opentelemetry setup with the main process.

        # Ready
        logger.info("LLMProxy preparation is done. Will start the server.")
        yield

        # Clean up

        logger.info("LLMProxy server is cleaning up.")

        # Remove worker config to avoid stale references.
        if self._config_file and os.path.exists(self._config_file):
            os.unlink(self._config_file)

        logger.info("LLMProxy server finishes.")

    async def start(self):
        """Start the proxy server thread and initialize global wiring.

        Side effects:

        * Sets the module-level global store for middleware/exporter access.
        * Calls `initialize()` once to register middleware and callbacks.
        * Writes a temporary YAML config consumed by LiteLLM worker.
        * Launches uvicorn in a daemon thread and waits for readiness.
        """
        # Refresh the serve context
        self.server_launcher.serve_context = self._serve_context()

        if self.store is None:
            raise ValueError("Store is not set. Please set the store before starting the LLMProxy.")

        store_capabilities = self.store.capabilities
        if self.server_launcher.args.launch_mode == "mp" and not store_capabilities.get("zero_copy", False):
            raise RuntimeError(
                "The store does not support zero-copy. Please use another store, or use asyncio or thread mode to launch the server."
            )
        elif self.server_launcher.args.launch_mode == "thread" and not store_capabilities.get("thread_safe", False):
            raise RuntimeError(
                "The store is not thread-safe. Please use another store, or use asyncio mode to launch the server."
            )
        elif self.server_launcher.args.launch_mode == "asyncio" and not store_capabilities.get("async_safe", False):
            raise RuntimeError("The store is not async-safe. Please use another store.")

        logger.info(
            f"Starting LLMProxy server in {self.server_launcher.args.launch_mode} mode with store capabilities: {store_capabilities}"
        )

        await self.server_launcher.start()

    async def stop(self):
        """Stop the proxy server and clean up temporary artifacts.

        This is a best-effort graceful shutdown with a bounded join timeout.
        """
        if not self.is_running():
            logger.warning("LLMProxy is not running. Nothing to stop.")
            return

        await self.server_launcher.stop()

    async def restart(self, *, _port: int | None = None) -> None:
        """Restart the proxy if running, else start it.

        Convenience wrapper calling `stop()` followed by `start()`.
        """
        logger.info("Restarting LLMProxy server...")
        if self.is_running():
            await self.stop()
        if _port is not None:
            self.server_launcher_args.port = _port
        await self.start()

    def is_running(self) -> bool:
        """Return whether the uvicorn server is active.

        Returns:
            bool: True if server was started and did not signal exit.
        """
        return self.server_launcher.is_running()

    def as_resource(
        self,
        rollout_id: str | None = None,
        attempt_id: str | None = None,
        model: str | None = None,
        sampling_parameters: Dict[str, Any] | None = None,
    ) -> LLM:
        """Create an `LLM` resource pointing at this proxy with rollout context.

        The returned endpoint is:
            `http://{host}:{port}/rollout/{rollout_id}/attempt/{attempt_id}`

        Args:
            rollout_id: Rollout identifier used for span attribution. If None, will instantiate a ProxyLLM resource.
            attempt_id: Attempt identifier used for span attribution. If None, will instantiate a ProxyLLM resource.
            model: Logical model name to use. If omitted and exactly one model
                is configured or all models have the same name, that model is used.
            sampling_parameters: Optional default sampling parameters.

        Returns:
            LLM: Configured resource ready for OpenAI-compatible calls.

        Raises:
            ValueError: If `model` is omitted and zero or multiple models are configured.
        """
        if model is None:
            if len(self.model_list) == 1:
                model = self.model_list[0]["model_name"]
            elif len(self.model_list) == 0:
                raise ValueError("No models found in model_list. Please specify the model.")
            else:
                first_model_name = self.model_list[0]["model_name"]
                if all(model_config["model_name"] == first_model_name for model_config in self.model_list):
                    model = first_model_name
                else:
                    raise ValueError(
                        f"Multiple models found in model_list: {self.model_list}. Please specify the model."
                    )

        if rollout_id is None and attempt_id is None:
            return ProxyLLM(
                endpoint=self.server_launcher.access_endpoint,
                model=model,
                sampling_parameters=dict(sampling_parameters or {}),
            )
        elif rollout_id is not None and attempt_id is not None:
            return LLM(
                endpoint=f"{self.server_launcher.access_endpoint}/rollout/{rollout_id}/attempt/{attempt_id}",
                model=model,
                sampling_parameters=dict(sampling_parameters or {}),
            )
        else:
            raise ValueError("Either rollout_id and attempt_id must be provided, or neither.")


_global_llm_proxy: Optional[LLMProxy] = None
_callbacks_before_litellm_start: Optional[List[Any]] = None


def get_active_llm_proxy() -> LLMProxy:
    """Get the current global LLMProxy instance.

    Returns:
        Optional[LLMProxy]: The current LLMProxy if set, else None.
    """
    if _global_llm_proxy is None:
        raise ValueError("Global LLMProxy is not set. Please call llm_proxy.start() first.")
    return _global_llm_proxy


def set_active_llm_proxy(proxy: LLMProxy) -> None:
    """Set the current global LLMProxy instance.

    Args:
        proxy: The LLMProxy instance to set as global.
    """
    global _global_llm_proxy
    _global_llm_proxy = proxy


def initialize_llm_callbacks(callback_classes: List[Type[CustomLogger]]) -> bool:
    """Restore `litellm.callbacks` to a state that is just initialized by agent-lightning.

    When litellm is restarted multiple times in the same process, more and more callbacks
    will be appended to `litellm.callbacks`, which may exceed the MAX_CALLBACKS limit.
    This function remembers the initial state of `litellm.callbacks` and always restore to that state.

    Args:
        callback_classes: List of callback classes to register.

    Returns:
        Whether the callbacks are initialized for the first time.
    """
    global _callbacks_before_litellm_start

    if _callbacks_before_litellm_start is None:
        litellm.callbacks.extend([cls() for cls in callback_classes])  # type: ignore
        _callbacks_before_litellm_start = [*litellm.callbacks]  # type: ignore
        return True

    else:
        # Put whatever is missing in the new callback classes to the existing callbacks.
        for cls in callback_classes:
            if not any(isinstance(cb, cls) for cb in _callbacks_before_litellm_start):
                logger.info(f"Adding missing callback {cls} to the existing callbacks.")
                _callbacks_before_litellm_start.append(cls())

    _reset_litellm_logging_callback_manager()

    if LightningOpenTelemetry in callback_classes:
        # Check if tracer provider is malformed due to global tracer clear in tests.
        if not _check_tracer_provider():
            logger.warning(
                "Global tracer provider might have been cleared outside. Re-initializing OpenTelemetry callback."
            )
            _callbacks_before_litellm_start = [
                cb for cb in _callbacks_before_litellm_start if not isinstance(cb, LightningOpenTelemetry)
            ] + [LightningOpenTelemetry()]
        else:
            logger.debug("Global tracer provider is valid. Reusing existing OpenTelemetry callback.")
    # Otherwise, we just skip the check for opentelemetry and use the existing callback.

    litellm.callbacks.clear()  # type: ignore
    litellm.callbacks.extend(_callbacks_before_litellm_start)  # type: ignore
    return False


def _check_tracer_provider() -> bool:
    """Check if the global tracer provider is properly initialized.

    We don't guarantee the tracer provider is our tracer provider.

    Returns:
        bool: True if the tracer provider is valid, else False.
    """
    if (
        hasattr(trace_api, "_TRACER_PROVIDER")
        and trace_api._TRACER_PROVIDER is not None  # pyright: ignore[reportPrivateUsage]
    ):
        return True
    return False
