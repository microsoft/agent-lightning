# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Dict, Generator, List, Optional, cast

import requests

from agentlightning.store.base import LightningStore
from agentlightning.types.tracer import OtelResource, Span, SpanContext, TraceStatus

from .base import Tracer

if TYPE_CHECKING:
    from weave.trace.call import Call
else:
    Call = Any  # type: ignore

logger = logging.getLogger(__name__)


class WeaveTracer(Tracer):
    """
    Tracer implementation using Weave for telemetry and trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context. It tracks:
    - Function/method calls
    - Input/Output data
    - Exceptions
    and logs them to Weave Cloud (W&B backend) or optionally bypasses the network for testing.

    Attributes:
        project_name: Name of the Weave project. Used to initialize the Weave client.
        _store: Optional LightningStore instance for storing collected spans.
        _tracing_enabled: Flag to enable/disable tracing at runtime. (weave op requires this)
        postprocess_output: Placeholder attribute observed by Weave SDK for op inputs. (weave op requires this)

        _loop: Dedicated asyncio event loop for background tasks.
        _loop_thread: Thread running the dedicated event loop.
        _loop_ready: Event to signal when the loop is ready.
    """

    def __init__(
        self, *, project_name: str | None = None, wandb_api_key: str | None = None, pass_weave_service: bool = True
    ):
        """
        Initialize a WeaveTracer instance.

        Args:
            project_name: Optional project name for Weave; defaults to the current module name.
            wandb_api_key: Optional W&B API key; sets environment variable if provided.
            pass_weave_service: Whether to bypass actual Weave/W&B network calls (for testing).
        """
        super().__init__()
        self.project_name = project_name or __name__
        self.sequence_id = 0
        self._store: Optional[LightningStore] = None
        self.pass_weave_service = pass_weave_service

        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

        # Attributes observed by Weave SDK for op inputs
        self._tracing_enabled = True
        self.postprocess_output = None

        # Private asyncio loop running in a daemon thread
        self._loop_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def init_worker(self, worker_id: int, store: Optional[LightningStore] = None):
        """
        Initialize the tracer for a worker thread/process.

        Args:
            worker_id: Identifier of the worker.
            store: Optional LightningStore for storing spans.
        """
        super().init_worker(worker_id, store)
        logger.info(f"[Worker {worker_id}] Setting up Weave tracer...")
        self._store = store

        try:
            import weave
        except ImportError:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.")

        # Optionally patch network calls to bypass real Weave/W&B endpoints
        if self.pass_weave_service:
            self.bypass_weave_service()

        # Initialize the Weave client if not already initialized
        if weave.get_client() is None:  # type: ignore
            try:
                weave.init(project_name=self.project_name)  # type: ignore
                logger.info(f"[Worker {worker_id}] Weave client initialized.")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Weave for project '{self.project_name}': {e}")

    def teardown_worker(self, worker_id: int):
        """
        Clean up tracer resources for the worker.

        Args:
            worker_id: Identifier of the worker.
        """
        super().teardown_worker(worker_id)
        self.shutdown()

    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Create an asynchronous tracing context using Weave.

        This context manager can be used with `async with` blocks to collect spans.

        Args:
            name: Optional name for the tracing context; defaults to the project name.
            store: Optional LightningStore to store spans.
            rollout_id: Optional rollout ID for trace context.
            attempt_id: Optional attempt ID for trace context.

        Yields:
            The Weave tracer instance for span collection.
        """
        with self._trace_context_sync(name=name, store=store, rollout_id=rollout_id, attempt_id=attempt_id):
            yield

    @contextmanager
    def _trace_context_sync(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> Generator[Any, None, None]:
        """
        Synchronous implementation of the tracing context.

        Args:
            name: Optional operation name.
            store: Optional LightningStore instance.
            rollout_id: Optional rollout ID.
            attempt_id: Optional attempt ID.

        Raises:
            ValueError: If store, rollout_id, and attempt_id are inconsistently provided.
            RuntimeError: If Weave is not installed or client is uninitialized.
        """
        arg_op = name or self.project_name
        arg_inputs: dict[str, str] | None = {"rollout_id": rollout_id or "", "attempt_id": attempt_id or ""}

        if store is not None and rollout_id is not None and attempt_id is not None:
            self._rollout_id = rollout_id
            self._attempt_id = attempt_id
            self._store = store
        else:
            raise ValueError("store, rollout_id, and attempt_id must be either all provided")

        try:
            import datetime

            import weave
        except ImportError:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.")

        weave_client = weave.get_client()  # type: ignore
        if not weave_client:
            raise RuntimeError("Weave client is not initialized. Call init_worker() first.")

        # Create a new trace call object in Weave
        trace_call = weave_client.create_call(op=arg_op, inputs=arg_inputs)  # type: ignore
        trace_call.started_at = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            yield
        except Exception as e:
            # Finish trace and log any exception
            weave_client.finish_call(trace_call, exception=e, op=self)  # type: ignore
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={e}")
        finally:
            # Finish trace even if no exception
            weave_client.finish_call(trace_call, op=self)  # type: ignore

    def bypass_weave_service(self):
        """
        Patch the Weave/W&B integration to bypass actual network calls for testing.

        - Mocks HTTP POST/GET requests
        - Patches wandb.Api methods
        - Silences Weave logging
        - Sets dummy WANDB_API_KEY if not provided
        """
        import weave
        from weave.compat import wandb # type: ignore

        _weave_tracer_entity_name = "weave_tracer_entity"

        def default_entity_name_getter(_self) -> str:  # type: ignore
            return _weave_tracer_entity_name

        def upsert_project_getter(
            _self, project: str, description: Optional[str] = None, entity: Optional[str] = None  # type: ignore
        ) -> dict[str, Any]:
            return {
                "upsertModel": {
                    "model": {
                        "name": project,
                        "description": description or "",
                        "entity": entity or _weave_tracer_entity_name,
                    }
                },
                "project": "weave_tracer_project",
            }

        # Mock network requests to avoid real HTTP calls
        def post(url: str, *args: Any, **kwargs: Any) -> requests.Response:
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"digest": "mocked_digest"}'
            return response

        def get(url: str, *args: Any, **kwargs: Any) -> requests.Response:
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"min_required_weave_python_version": "0.52.14"}'
            return response

        # Patch API methods and HTTP requests
        wandb.Api.default_entity_name = default_entity_name_getter  # type: ignore
        wandb.Api.upsert_project = upsert_project_getter  # type: ignore
        weave.utils.http_requests.session.post = post  # type: ignore
        weave.utils.http_requests.session.get = get  # type: ignore

        # Silence Weave logging
        for name in logging.root.manager.loggerDict:
            if name.startswith("weave"):
                logging.getLogger(name).disabled = True

        # Set dummy API key if missing
        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = "dumped_api_key_for_weave_tracer"

        # if needed in future tests, enable this and replace WF_TRACE_SERVER_URL to local server
        # full_url = f"http://127.0.0.1:{_port}"
        # os.environ["WF_TRACE_SERVER_URL"] = full_url

    def _ensure_loop(self) -> None:
        """
        Ensure that the dedicated asyncio event loop is running in a background daemon thread.
        """
        if self._loop_thread is None or self._loop is None:
            self._loop_ready.clear()
            self._loop_thread = threading.Thread(target=self._loop_runner, name="otel-loop", daemon=True)
            self._loop_thread.start()
            self._loop_ready.wait()  # Wait until the loop is ready

    def _await_in_loop(self, coro: Awaitable[Any], timeout: Optional[float] = None) -> Any:
        """
        Submit a coroutine to the dedicated loop and wait synchronously for the result.

        Args:
            coro: Coroutine to execute.
            timeout: Optional timeout in seconds.

        Returns:
            Result of the coroutine execution.
        """
        self._ensure_loop()
        if self._loop is None:
            raise RuntimeError("Loop is not initialized. This should not happen.")

        if threading.current_thread() is self._loop_thread:
            self._loop.call_soon_threadsafe(asyncio.create_task, coro)  # type: ignore
            return None

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore
        return fut.result(timeout=timeout)  # type: ignore

    def _loop_runner(self):
        """
        Target function for the background thread running the dedicated asyncio loop.
        """
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        loop.run_forever()
        loop.close()

    def shutdown(self) -> None:
        """Stop and clean up the dedicated asyncio loop and thread."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
        if self._loop_thread:
            self._loop_thread.join(timeout=5)

    def _on_finish_handler(self, call: "Call", *args: Any, **kwargs: Any) -> None: # type: ignore
        """
        Handler called when a Weave Call finishes.

        Converts the call (including nested children) into spans and stores them in LightningStore.
        """
        spans, self.sequence_id = self.convert_call_to_spans(call, self._rollout_id, self._attempt_id, self.sequence_id) # type: ignore

        if self._store and self._rollout_id and self._attempt_id:
            try:
                for span in spans:
                    self._ensure_loop()
                    self._await_in_loop(
                        self._store.add_span(span),
                        timeout=60.0,
                    )
            except Exception as e:
                logger.exception(f"Error adding span to store: {e}")

    def convert_call_to_spans(
        self,
        call: "Call", # type: ignore
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        seq_start: int = 0,
    ) -> tuple[List[Span], int]:
        """
        Recursively convert a Weave Call (with nested children) into a flat list of Agent Lightning Spans.

        Args:
            call: The Weave Call object.
            rollout_id: Optional rollout ID to attach to spans.
            attempt_id: Optional attempt ID to attach to spans.
            seq_start: Sequence number to start from.

        Returns:
            Tuple of (list_of_spans, next_sequence_id).
        """
        from collections.abc import Sequence

        spans: List[Span] = []
        sequence_id = seq_start

        rollout_id = rollout_id or getattr(call, "inputs", {}).get("rollout_id", "")
        attempt_id = attempt_id or getattr(call, "inputs", {}).get("attempt_id", "")

        start_dt = getattr(call, "started_at", None)
        start_ts: Optional[float] = start_dt.timestamp() if start_dt else None

        end_dt = getattr(call, "ended_at", None)
        end_ts: Optional[float] = end_dt.timestamp() if end_dt else None

        attributes = dict(getattr(call, "attributes", {}) or {})
        flat_attrs: Dict[str, Any] = {}

        # Flatten nested attribute dictionaries
        for k, v in attributes.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():  # type: ignore
                    sub_v = cast(Any, sub_v)
                    if isinstance(sub_v, (str, bool, int, float)) or isinstance(sub_v, Sequence):
                        flat_attrs[f"{k}.{sub_k}"] = sub_v
                    else:
                        flat_attrs[f"{k}.{sub_k}"] = str(sub_v)
            else:
                flat_attrs[k] = v

        trace_id = str(getattr(call, "trace_id", None))
        span_id = str(getattr(call, "id", None))
        parent_id = str(getattr(call, "parent_id", None)) if getattr(call, "parent_id", None) else None

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_state={},
        )

        parent_context = (
            SpanContext(
                trace_id=trace_id,
                span_id=parent_id,
                is_remote=False,
                trace_state={},
            )
            if parent_id
            else None
        )

        # Build the Span object
        span = Span(
            rollout_id=rollout_id or "",
            attempt_id=attempt_id or "",
            sequence_id=sequence_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=getattr(call, "func_name", "unknown"),
            status=TraceStatus(status_code="OK"),
            attributes=flat_attrs,
            events=[],  # Weave calls do not generate events
            links=[],  # Weave calls do not generate links
            start_time=start_ts,
            end_time=end_ts,
            context=context,
            parent=parent_context,
            resource=OtelResource(attributes={}, schema_url=""),
        )

        spans.append(span)
        sequence_id += 1

        children: List["Call"] = getattr(call, "_children", []) # type: ignore
        # Recursively process child calls
        for child in children:
            child_spans, sequence_id = self.convert_call_to_spans( # type: ignore
                child,
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                seq_start=sequence_id,
            )
            spans.extend(child_spans)

        return spans, sequence_id
