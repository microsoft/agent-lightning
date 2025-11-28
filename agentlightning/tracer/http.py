# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import multiprocessing
import queue
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from httpdbg.hooks.all import httprecord
from httpdbg.records import HTTPRecords
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.span import (
    SpanContext,
    TraceFlags,
    TraceState,
)

from agentlightning.store import LightningStore

from .base import Tracer

logger = logging.getLogger(__name__)


class HttpTracer(Tracer):
    """
    A tracer implementation that captures HTTP requests using httpdbg.

    This tracer hooks into the Python HTTP libraries and captures all
    HTTP requests and responses made during the traced code execution.
    The captured requests are converted to OpenTelemetry spans for
    compatibility with the rest of the tracing ecosystem.

    Caution: The current implementation of HttpTracer is very fragile,
    and we do not recommend using it in production.
    It is primarily for demonstration and testing purposes.

    Deprecated: This tracer is deprecated and will be removed in a future version.
    Please use LLMProxy as an alternative.

    Attributes:
        include_headers: Whether to include HTTP headers in the spans.
            Headers may contain sensitive information. Use with caution.
        include_body: Whether to include HTTP request and response bodies in the spans.
            Bodies may be large and contain sensitive information. Use with caution.
        include_agentlightning_requests: Whether to include requests initiated by AgentLightning itself.
        subprocess_mode: Whether to run trace_run and trace_run_async in subprocesses for isolation.
        subprocess_timeout: Timeout for subprocess execution in seconds.
    """

    AGENTLIGHTNING_HEADERS = {"x-agentlightning-client"}

    def __init__(
        self,
        include_headers: bool = False,
        include_body: bool = False,
        include_agentlightning_requests: bool = False,
        subprocess_mode: bool = True,
        subprocess_timeout: float = 3600.0,
    ):
        super().__init__()
        self._last_records: Optional[HTTPRecords] = None
        self.include_headers = include_headers
        self.include_body = include_body
        self.include_agentlightning_requests = include_agentlightning_requests
        self.subprocess_mode = subprocess_mode
        self.subprocess_timeout = subprocess_timeout

    def init_worker(self, worker_id: int, store: Optional[LightningStore] = None) -> None:
        """
        Initialize the tracer in a worker process.

        Args:
            worker_id: The ID of the worker process.
            store: The store to add the spans to.
        """
        super().init_worker(worker_id, store)
        logger.info(f"[Worker {worker_id}] HttpTracer initialized.")

    @asynccontextmanager
    async def trace_context(self, name: Optional[str] = None, **kwargs: Any) -> AsyncGenerator[HTTPRecords, None]:
        """
        Starts a new HTTP tracing context. This should be used as a context manager.

        Args:
            name: Optional name for the tracing context.
        """
        with self._trace_context_sync(name=name, **kwargs) as records:
            yield records

    @contextmanager
    def _trace_context_sync(self, name: Optional[str] = None, **kwargs: Any) -> Iterator[HTTPRecords]:
        """
        Starts a new HTTP tracing context. This should be used as a context manager.

        Args:
            name: Optional name for the tracing context.

        Yields:
            The HTTPRecords instance containing traced HTTP activities.
        """
        records = HTTPRecords()
        with httprecord(records):
            self._last_records = records
            yield records

    def get_last_trace(self) -> List[ReadableSpan]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects converted from HTTP records.
        """
        if self._last_records is None:
            return []

        return self._convert_to_spans(self._last_records)

    def _convert_to_spans(self, records: HTTPRecords) -> List[ReadableSpan]:
        """
        Convert HTTPRecords to OpenTelemetry spans.

        Args:
            records: The HTTPRecords instance containing HTTP traces.

        Returns:
            A list of ReadableSpan objects representing the HTTP activities.
        """
        spans: List[ReadableSpan] = []

        # Create a trace ID that will be shared by all spans in this trace
        trace_id = int(uuid.uuid4().hex[:16], 16)

        for record in records.requests.values():
            # Skip AgentLightning requests if include_agentlightning_requests is False
            should_skip = False
            if not self.include_agentlightning_requests and record.request and record.request.headers:
                for header in record.request.headers:
                    if header.name.lower() in self.AGENTLIGHTNING_HEADERS and header.value.lower() == "true":
                        should_skip = True
                        break

            if should_skip:
                continue

            # Create a span ID for this specific HTTP request
            span_id = int(uuid.uuid4().hex[:8], 16)

            # Create a span context
            span_context = SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                trace_state=TraceState(),
            )

            # Extract important information from the HTTP record
            method = record.method
            url = record.url
            parsed_url = urlparse(url)
            status_code = record.status_code

            # Create attributes dictionary
            attributes: Dict[str, Any] = {
                "http.method": method,
                "http.url": url,
                "http.target": parsed_url.path,
                "http.host": parsed_url.netloc,
            }

            if status_code is not None and status_code > 0:  # type: ignore
                attributes["http.status_code"] = status_code

            # Calculate duration - from begin time to last update
            duration = None
            if hasattr(record, "last_update") and record.last_update and record.tbegin:
                duration = (record.last_update - record.tbegin).total_seconds()
                attributes["http.duration_ms"] = duration * 1000  # Convert to ms

            # Optionally include headers
            if self.include_headers and record.request and record.request.headers:
                for header in record.request.headers:
                    header_name = header.name.lower()
                    attributes[f"http.request.header.{header_name}"] = header.value

            if self.include_headers and record.response and record.response.headers:
                for header in record.response.headers:
                    header_name = header.name.lower()
                    attributes[f"http.response.header.{header_name}"] = header.value

            # Optionally include body - preserve complete content for analysis
            if self.include_body and record.request:
                body_content = record.request.content
                if body_content:
                    # Store raw body content for later parsing/analysis
                    attributes["http.request.body"] = body_content

            if self.include_body and record.response:
                body_content = record.response.content
                if body_content:
                    # Store raw body content for later parsing/analysis
                    attributes["http.response.body"] = body_content

            # Determine span status
            span_status = StatusCode.OK
            if status_code and status_code >= 400 or record.exception:
                span_status = StatusCode.ERROR

            # Create start and end timestamps in nanoseconds
            # If we have duration, use it, otherwise default to current time - 1ms
            start_time_ns = int(record.tbegin.timestamp() * 1e9)
            if duration:
                end_time_ns = int((record.tbegin.timestamp() + duration) * 1e9)
            else:
                end_time_ns = int(record.last_update.timestamp() * 1e9)

            span = ReadableSpan(
                name=f"HTTP {method} {url}",
                context=span_context,
                parent=None,
                kind=SpanKind.CLIENT,
                status=Status(span_status),
                start_time=start_time_ns,
                end_time=end_time_ns,
                attributes=attributes,
                events=[],
                links=[],
                resource=None,
            )

            spans.append(span)

        return spans

    def trace_run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        A convenience wrapper to trace the execution of a single synchronous function.

        If subprocess_mode is enabled, the function will be executed in an isolated subprocess
        to prevent HTTP hooks from affecting the parent process.

        Args:
            func: The synchronous function to execute and trace.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the function.
        """
        if self.subprocess_mode:
            return self._trace_run_subprocess(func, args, kwargs)
        else:
            return super().trace_run(func, *args, **kwargs)

    async def trace_run_async(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """
        A convenience wrapper to trace the execution of a single asynchronous function.

        If subprocess_mode is enabled, the function will be executed in an isolated subprocess
        to prevent HTTP hooks from affecting the parent process.

        Args:
            func: The asynchronous function to execute and trace.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the function.
        """
        if self.subprocess_mode:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._trace_run_subprocess, func, args, kwargs, True  # True for async
            )
        else:
            return await super().trace_run_async(func, *args, **kwargs)

    def _trace_run_subprocess(
        self,
        func: Callable[..., Any],
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        is_async: bool = False,
    ) -> Any:
        """
        Execute a function in a subprocess with HTTP tracing.

        Args:
            func: The function to execute.
            args: Positional arguments to pass to the function.
            kwargs: Keyword arguments to pass to the function.
            is_async: Whether the function is asynchronous.

        Returns:
            The return value of the function.
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        # Create a queue to receive results from the subprocess
        result_queue = multiprocessing.Queue()  # type: ignore

        # Create and start the subprocess
        process = multiprocessing.Process(
            target=self._subprocess_worker, args=(func, args, kwargs, result_queue, is_async)  # type: ignore
        )
        process.start()

        try:
            # Wait for the process to complete and get the result
            process.join(timeout=self.subprocess_timeout)
            result = result_queue.get_nowait()  # type: ignore

            if result["success"]:
                # Store the captured records for get_last_trace()
                self._last_records = result["records"]
                return result["return_value"]  # type: ignore
            else:
                if "records" in result:
                    self._last_records = result["records"]
                # Re-raise the exception that occurred in the subprocess
                raise result["exception"]

        except multiprocessing.TimeoutError:
            process.terminate()
            process.join()
            raise TimeoutError(f"Subprocess execution timed out after {self.subprocess_timeout} seconds.")
        except queue.Empty:
            logger.error("Traced result is empty. This may indicate a timeout or an issue with the subprocess.")
        finally:
            if process.is_alive():
                process.terminate()
                process.join()

    def _subprocess_worker(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        result_queue: multiprocessing.Queue,  # type: ignore
        is_async: bool,
    ) -> None:
        """
        Worker function that runs in the subprocess to execute the traced function.

        Args:
            func: The function to execute.
            args: Positional arguments.
            kwargs: Keyword arguments.
            result_queue: Queue to send results back to parent process.
            is_async: Whether the function is asynchronous.
        """
        # Create a new tracer instance in the subprocess (without subprocess mode to avoid recursion)
        subprocess_tracer = HttpTracer(
            include_headers=self.include_headers,
            include_body=self.include_body,
            include_agentlightning_requests=self.include_agentlightning_requests,
            subprocess_mode=False,  # Disable subprocess mode in the worker
        )

        try:
            if is_async:
                # Run async function in new event loop
                import asyncio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return_value = loop.run_until_complete(subprocess_tracer.trace_run_async(func, *args, **kwargs))
                finally:
                    loop.close()
            else:
                # Run sync function
                return_value = subprocess_tracer.trace_run(func, *args, **kwargs)

            # Get the captured records
            records = subprocess_tracer._last_records

            # Send success result back to parent
            result_queue.put({"success": True, "return_value": return_value, "records": records})  # type: ignore

        except Exception as e:
            # Log the exception
            logger.exception(f"Error in subprocess worker in http tracer: {e}")

            # Get the captured records even when there's an exception
            records = subprocess_tracer._last_records
            # Send error result back to parent
            result_queue.put({"success": False, "exception": e, "records": records})  # type: ignore
