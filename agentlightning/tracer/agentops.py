from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import List, Optional, TYPE_CHECKING
import uuid

import agentops.sdk.core
import agentops
from agentops.sdk.core import TracingCore
from agentops.sdk.processors import SpanProcessor
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.instrumentation.agentops import AgentOpsServerManager, set_switch
from agentlightning.instrumentation import instrument_all, uninstrument_all
from agentlightning.types import Task
from .base import BaseTracer


if TYPE_CHECKING:
    from agentops.integration.callbacks.langchain import LangchainCallbackHandler


logger = logging.getLogger(__name__)


class AgentOpsTracer(BaseTracer):
    """Traces agent execution using AgentOps.

    This tracer provides functionality to capture execution details using the
    AgentOps library. It manages the AgentOps client initialization, server setup,
    and integration with the OpenTelemetry tracing ecosystem.

    Attributes:
        agentops_managed: Whether to automatically manage `agentops`.
                          When set to true, tracer calls `agentops.init()`
                          automatically and launches an agentops endpoint locally.
                          If not, you are responsible for calling and using it
                          before using the tracer.
        instrument_managed: Whether to automatically manage instrumentation.
                            When set to false, you will manage the instrumentation
                            yourself and the tracer might not work as expected.
        daemon: Whether the AgentOps server runs as a daemon process.
                Only applicable if `agentops_managed` is True.
        upload_every_n_tasks: Number of tasks between uploads to AgentOps.
                              `AGENTOPS_API_KEY` must be set in the environment.
        upload_every_n_tasks_trained: Number of tasks between uploads to AgentOps
                                      when the agent is in training mode. You need
                                      `AGENTOPS_API_KEY` set to use this feature.
    """

    def __init__(
        self,
        *,
        agentops_managed: bool = True,
        instrument_managed: bool = True,
        daemon: bool = True,
        upload_every_n_tasks: int | None = None,
        upload_every_n_tasks_trained: int | None = None,
    ):
        super().__init__()
        self._lightning_span_processor: Optional[LightningSpanProcessor] = None
        self.agentops_managed = agentops_managed
        self.instrument_managed = instrument_managed
        self.daemon = daemon
        self.upload_every_n_tasks = upload_every_n_tasks
        self.upload_every_n_tasks_trained = upload_every_n_tasks_trained

        self._agentops_server_manager = AgentOpsServerManager(self.daemon)
        self._agentops_server_port_val: Optional[int] = None
        self._uploading_state: bool | None = None

        self._original_provider = None
        self._original_meter_provider = None

        if not self.agentops_managed:
            logger.warning("agentops_managed=False. You are responsible for AgentOps setup.")
        if not self.instrument_managed:
            logger.warning("instrument_managed=False. You are responsible for all instrumentation.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_agentops_server_manager"] = None  # Exclude the unpicklable server manager
        # _agentops_server_port_val (int) is inherently picklable and will be included.
        logger.debug(f"Getting state for pickling Trainer (PID {os.getpid()}). _agentops_server_manager excluded.")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # In child process, self._agentops_server_manager will be None.
        logger.debug(f"Setting state for unpickled Trainer (PID {os.getpid()}). _agentops_server_manager is None.")

    def init(self, *args, **kwargs):
        if self.agentops_managed and self._agentops_server_manager:
            self._agentops_server_manager.start()
            self._agentops_server_port_val = self._agentops_server_manager.get_port()
            if self._agentops_server_port_val is None:
                if (
                    self._agentops_server_manager.server_process is not None
                    and self._agentops_server_manager.server_process.is_alive()
                ):
                    raise RuntimeError("AgentOps server started but port is None. Check server manager logic.")
                elif (
                    self._agentops_server_port_val is None and self._agentops_server_manager.server_process is None
                ):  # Server failed to start
                    raise RuntimeError("AgentOps server manager indicates server is not running and port is None.")

    def teardown(self):
        if self.agentops_managed:
            self._agentops_server_manager.stop()
            logger.info("AgentOps server stopped.")

    def instrument(self, worker_id: int):
        instrument_all()

    def uninstrument(self, worker_id: int):
        uninstrument_all()

    def init_worker(self, worker_id: int):
        super().init_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Setting up tracer...")  # worker_id included in process name

        if self.instrument_managed:
            self.instrument(worker_id)
            logger.info(f"[Worker {worker_id}] Instrumentation applied.")

        if self.agentops_managed:
            if "AGENTOPS_API_KEY" not in os.environ:
                logger.warning(
                    f"[Worker {self.worker_id}] AGENTOPS_API_KEY environment variable is not set. "
                    "Using local non-op server for tracing."
                )
                api_key = str(uuid.uuid4())  # Use a dummy key for local testing
            else:
                logger.info(f"[Worker {self.worker_id}] AgentOps API key found in environment.")
                api_key = os.environ["AGENTOPS_API_KEY"]

            agentops.init(api_key=api_key, auto_start_session=False)

        self._lightning_span_processor = LightningSpanProcessor()

        try:
            # new versions
            instance = agentops.sdk.core.tracer
            instance.provider.add_span_processor(self._lightning_span_processor)
        except AttributeError:
            # old versions
            instance = TracingCore.get_instance()
            instance._provider.add_span_processor(self._lightning_span_processor)

    def teardown_worker(self, worker_id: int) -> None:
        super().teardown_worker(worker_id)

        if self.instrument_managed:
            self.uninstrument(worker_id)
            logger.info(f"[Worker {worker_id}] Instrumentation removed.")

    @contextmanager
    def trace_context(self, name: Optional[str] = None, task: Optional[Task] = None, **kwargs):
        """
        Starts a new tracing context. This should be used as a context manager.

        Args:
            name: Optional name for the tracing context.

        Yields:
            The LightningSpanProcessor instance to collect spans.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")

        task_index = task.task_index if task and task.task_index is not None else None
        task_mode = task.mode if task else "unknown"
        online_tracing = task_index is not None and (
            (
                task_mode == "train"
                and self.upload_every_n_tasks_trained is not None
                and task_index % self.upload_every_n_tasks_trained == 0
            )
            or (
                task_mode in ["val", "test"]
                and self.upload_every_n_tasks is not None
                and task_index % self.upload_every_n_tasks == 0
            )
        )
        if online_tracing:
            logger.info(
                f"[Worker {self.worker_id}] AgentOps online tracing "
                f"for task {task_index if task_index is not None else 'unknown'}."
            )
            set_switch(True)
        else:
            set_switch(False)

        end_state = "Success"
        end_state_reason = None
        trace_context = None
        try:
            trace_context = agentops.start_trace(name or "session", tags=[f"{task_mode}_task_{task_index or 0:04d}"])
            with self._lightning_span_processor:
                yield self._lightning_span_processor
        except Exception as e:
            end_state = "Error"
            end_state_reason = str(e)
            raise
        finally:
            if trace_context is None:
                logger.error(
                    f"[Worker {self.worker_id}] Trace context is None. AgentOps might not be initialized properly."
                )
            agentops.end_trace(trace_context=trace_context, end_state=end_state)
            logger.info(
                f"[Worker {self.worker_id}] AgentOps trace ended with state: {end_state}, reason: {end_state_reason}"
            )

    def get_last_trace(self) -> List[ReadableSpan]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")
        return self._lightning_span_processor.spans()

    def get_langchain_callback_handler(self, tags: List[str] | None = None) -> LangchainCallbackHandler:
        """
        Get the Langchain callback handler for integrating with Langchain.

        Args:
            tags: Optional list of tags to apply to the Langchain callback handler.

        Returns:
            An instance of the Langchain callback handler.
        """
        import agentops
        from agentops.integration.callbacks.langchain import LangchainCallbackHandler

        tags = tags or []
        client_instance = agentops.get_client()
        api_key = None
        if client_instance.initialized:
            api_key = client_instance.config.api_key
        else:
            logger.warning(
                "AgentOps client not initialized when creating LangchainCallbackHandler. API key may be missing."
            )
        return LangchainCallbackHandler(api_key=api_key, tags=tags)


class LightningSpanProcessor(SpanProcessor):

    _spans: List[ReadableSpan] = []

    def __enter__(self):
        self._last_trace = None
        self._spans = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def spans(self) -> List[ReadableSpan]:
        """
        Get the list of spans collected by this processor.
        This is useful for debugging and testing purposes.

        Returns:
            List of ReadableSpan objects collected during tracing.
        """
        return self._spans

    def on_end(self, span: ReadableSpan) -> None:
        """
        Process a span when it ends.

        Args:
            span: The span that has ended.
        """
        # Skip if span is not sampled
        if not span.context or not span.context.trace_flags.sampled:
            return

        self._spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
