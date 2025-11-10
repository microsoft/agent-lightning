# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import socket
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, TypedDict, Union, cast

import litellm
import opentelemetry.trace as trace_api
import uvicorn
import yaml
from fastapi import Request, Response
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.proxy.proxy_server import app, save_worker_config  # pyright: ignore[reportUnknownVariableType]
from litellm.types.utils import CallTypes
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from starlette.middleware.base import BaseHTTPMiddleware

from agentlightning.types import LLM, ProxyLLM

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
        self._lock: Optional[threading.RLock] = None
        self._loop_lock_pid: Optional[int] = None

        # Single dedicated event loop running in a daemon thread.
        # This decouples OTEL SDK threads from our async store I/O.
        # Deferred creation until first use.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

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

    def _ensure_lock(self) -> threading.RLock:
        """Lazily initialize the lock on first use.

        Returns:
            threading.RLock: The initialized lock.
        """
        self._clear_loop_and_lock()
        if self._lock is None:
            self._lock = threading.RLock()
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
        import traceback

        traceback.print_stack()
        # Buffer append under lock to protect against concurrent exporters.
        with self._ensure_lock():
            for span in spans:
                print(
                    span.name,
                    "Span =",
                    span.get_span_context().span_id,
                    "Parent =",
                    span.parent.span_id if span.parent else None,
                    "Start =",
                    span.start_time,
                    "End =",
                    span.end_time,
                )
                self._buffer.append(span)

        # Run the async flush on our private loop, synchronously from caller's POV.
        async def _locked_flush():
            # Take the lock inside the coroutine to serialize with other flushes.
            with self._ensure_lock():
                return await self._maybe_flush()

        try:
            loop = self._ensure_loop()
            fut = asyncio.run_coroutine_threadsafe(_locked_flush(), loop)
            fut.result()  # Bubble up any exceptions from the coroutine.
        except Exception as e:
            logger.exception("Export flush failed: %s", e)
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    async def _maybe_flush(self):
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

            store = self._store or get_active_llm_proxy().get_store()
            if store is None:
                logger.warning("Store is not set in LLMProxy. Cannot log spans to store.")
                continue

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
                logger.warning(f"No headers found in {len(subtree_spans)} subtree spans. Cannot log to store.")
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
            for span in subtree_spans:
                await store.add_otel_span(
                    rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id_decimal, readable_span=span
                )

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
        print("kwargs keys =", kwargs.keys())
        print("call type =", call_type)
        if "metadata" not in kwargs or "litellm_parent_otel_span" not in kwargs["metadata"]:
            parent_otel_span = self.create_litellm_proxy_request_started_span(  # type: ignore
                start_time=datetime.now(),
                headers=kwargs.get("headers", {}),
            )
            updated_metadata = {**kwargs.get("metadata", {}), "litellm_parent_otel_span": parent_otel_span}
            print("updated_metadata =", updated_metadata)
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

    Usage Note:
    As the LLM Proxy sets up an OpenTelemetry tracer, it's recommended to run it in a different
    process from the main runner (i.e., tracer from agents).

    !!! warning

        The LLM Proxy does support streaming, but the tracing is still problematic when streaming is enabled.

    !!! danger

        Do not run LLM proxy in the same process as the main runner. It's easy to cause conflicts in the tracer provider
        with tracers like [`AgentOpsTracer`][agentlightning.AgentOpsTracer].

    Args:
        port: TCP port to bind.
        model_list: LiteLLM `model_list` entries.
        store: LightningStore used for span sequence and persistence.
        host: Publicly reachable host used in resource endpoints. Defaults to best-guess IPv4.
        litellm_config: Extra LiteLLM proxy config merged with `model_list`.
        num_retries: Default LiteLLM retry count injected into `litellm_settings`.
    """

    def __init__(
        self,
        port: int,
        model_list: List[ModelConfig] | None = None,
        store: Optional[LightningStore] = None,
        host: str | None = None,
        litellm_config: Dict[str, Any] | None = None,
        num_retries: int = 0,
        _add_return_token_ids: bool = True,
    ):
        self.store = store
        self.host = host or _get_default_ipv4_address()
        self.port = port
        self.model_list = model_list or []
        self.litellm_config = litellm_config or {}

        # Ensure num_retries is present inside the litellm_settings block.
        self.litellm_config.setdefault("litellm_settings", {})
        self.litellm_config["litellm_settings"].setdefault("num_retries", num_retries)

        self._server_thread = None
        self._config_file = None
        self._uvicorn_server = None
        self._ready_event = threading.Event()

        self._add_return_token_ids = _add_return_token_ids

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
        """Replace the in-memory model list and hot-restart if running.

        Args:
            model_list: New list of model entries.
        """
        self.model_list = model_list
        logger.info(f"Updating LLMProxy model list to: {model_list}")
        if self.is_running():
            self.restart()
        # Do nothing if the server is not running.

    def update_port(self, port: int) -> None:
        """Update the port for the proxy.

        Args:
            port: The new port to use for the proxy.
        """
        self.port = port

    def _wait_until_started(self, startup_timeout: float = 20.0):
        """Block until the uvicorn server reports started or timeout.

        Args:
            startup_timeout: Maximum seconds to wait.
        """
        start = time.time()
        while True:
            if self._uvicorn_server is None:
                break
            if self._uvicorn_server.started:
                self._ready_event.set()
                break
            if self._uvicorn_server.should_exit:
                break
            if time.time() - start > startup_timeout:
                break
            time.sleep(0.01)

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
        installed: bool = False
        for mw in app.user_middleware:
            if mw.cls is RolloutAttemptMiddleware:
                # Check whether the middleware is installed.
                # It could be installed by other LLM Proxy instances, but it doesn't matter.
                logger.info("Found existing RolloutAttemptMiddleware installed. Will not install a new one.")
                installed = True
                break

        if not installed:
            # Fallback to adding a new middleware
            logger.info("Adding a new middleware to the FastAPI app.")
            app.add_middleware(RolloutAttemptMiddleware)

        if not initialize_llm_callbacks(self._add_return_token_ids):
            # If it's not the first time to initialize the callbacks, also
            # reset LiteLLM's logging worker so its asyncio.Queue binds to the new loop.
            _reset_litellm_logging_worker()

    def start(self):
        """Start the proxy server thread and initialize global wiring.

        Side effects:

        * Sets the module-level global store for middleware/exporter access.
        * Calls `initialize()` once to register middleware and callbacks.
        * Writes a temporary YAML config consumed by LiteLLM worker.
        * Launches uvicorn in a daemon thread and waits for readiness.
        """
        if self.is_running():
            # Trigger restart
            self.stop()

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

        # Bind to all interfaces to allow other hosts to reach it if needed.
        self._uvicorn_server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=self.port))

        def run_server():
            # Serve uvicorn in this background thread with its own event loop.
            assert self._uvicorn_server is not None
            asyncio.run(self._uvicorn_server.serve())

        logger.info("Starting LLMProxy server thread...")
        self._ready_event.clear()
        # FIXME: This thread should either be reused or the whole proxy should live in another process.
        # Problem 1: in litellm worker, <Queue at 0x70f1d028cd90 maxsize=50000> is bound to a different event loop
        # Problem 2: Proxy has conflicted opentelemetry setup with the main process.
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._wait_until_started()

    def stop(self):
        """Stop the proxy server and clean up temporary artifacts.

        This is a best-effort graceful shutdown with a bounded join timeout.
        """
        if not self.is_running():
            logger.warning("LLMProxy is not running. Nothing to stop.")
            return

        # Remove worker config to avoid stale references.
        if self._config_file and os.path.exists(self._config_file):
            os.unlink(self._config_file)

        logger.info("Stopping LLMProxy server thread...")
        stop_success = True
        if self._server_thread is not None and self._uvicorn_server is not None and self._uvicorn_server.started:
            self._uvicorn_server.should_exit = True
            self._server_thread.join(timeout=10.0)  # Allow time for graceful shutdown.
            if self._server_thread.is_alive():
                logger.error(
                    "LLMProxy server thread is still alive after 10 seconds. Cannot kill it because it's a thread."
                )
                stop_success = False
            self._server_thread = None
            self._uvicorn_server = None
            self._config_file = None
            self._ready_event.clear()
            if not _check_port(self.host, self.port):
                logger.error(f"Port {self.port} is still in use. Stopping LLMProxy is not successful.")
                stop_success = False
        if stop_success:
            logger.info("LLMProxy server thread stopped.")
        else:
            logger.error("LLMProxy server is not stopped successfully.")

    def restart(self, *, _port: int | None = None) -> None:
        """Restart the proxy if running, else start it.

        Convenience wrapper calling `stop()` followed by `start()`.
        """
        logger.info("Restarting LLMProxy server...")
        if self.is_running():
            self.stop()
        if _port is not None:
            self.port = _port
        self.start()

    def is_running(self) -> bool:
        """Return whether the uvicorn server is active.

        Returns:
            bool: True if server was started and did not signal exit.
        """
        return self._uvicorn_server is not None and self._uvicorn_server.started

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
                endpoint=f"http://{self.host}:{self.port}",
                model=model,
                sampling_parameters=dict(sampling_parameters or {}),
            )
        elif rollout_id is not None and attempt_id is not None:
            return LLM(
                endpoint=f"http://{self.host}:{self.port}/rollout/{rollout_id}/attempt/{attempt_id}",
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


def initialize_llm_callbacks(_add_return_token_ids: bool = True) -> bool:
    """Restore `litellm.callbacks` to a state that is just initialized by agent-lightning.

    When litellm is restarted multiple times in the same process, more and more callbacks
    will be appended to `litellm.callbacks`, which may exceed the MAX_CALLBACKS limit.
    This function remembers the initial state of `litellm.callbacks` and always restore to that state.

    Args:
        _add_return_token_ids: Whether to add the return token ids callback. Internal use only.
        Ideally the callback should automatically be enabled when the backend supports it.

    Returns:
        Whether the callbacks are initialized for the first time.
    """
    global _callbacks_before_litellm_start

    if _callbacks_before_litellm_start is None:
        litellm.callbacks.extend(  # type: ignore
            [
                AddReturnTokenIds(),
                LightningOpenTelemetry(),
            ]
            if _add_return_token_ids
            else [
                LightningOpenTelemetry(),
            ]
        )
        _callbacks_before_litellm_start = [*litellm.callbacks]  # type: ignore
        return True

    _reset_litellm_logging_callback_manager()

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

    litellm.callbacks.clear()  # type: ignore
    litellm.callbacks.extend(_callbacks_before_litellm_start)  # type: ignore
    return False


def _get_default_ipv4_address() -> str:
    """Determine the default outbound IPv4 address for this machine.

    Implementation:
        Opens a UDP socket and "connects" to a public address to force route
        selection, then inspects the socket's local address. No packets are sent.

    Returns:
        str: Best-guess IPv4 like `192.168.x.y`. Falls back to `127.0.0.1`.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually contact 8.8.8.8; just forces the OS to pick a route.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _check_port(host: str, port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        result = s.connect_ex((host, port))
        return result != 0  # True if unavailable


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
