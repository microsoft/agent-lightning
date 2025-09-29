from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypedDict, Union, cast

import litellm
import uvicorn
import yaml
from fastapi import FastAPI, Request
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.proxy.proxy_server import app, save_worker_config  # pyright: ignore[reportUnknownVariableType]
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import LLM

from .store.base import LightningStore

logger = logging.getLogger(__name__)


class ModelConfig(TypedDict):
    """Model configuration in the format of LiteLLM's model_list."""

    model_name: str
    litellm_params: Dict[str, Any]


def _get_pre_call_data(args: Any, kwargs: Any) -> Dict[str, Any]:
    if kwargs.get("data"):
        data = kwargs["data"]
    elif len(args) >= 3:
        data = args[2]
    else:
        raise ValueError(f"Unable to get request data from args or kwargs: {args}, {kwargs}")
    if not isinstance(data, dict):
        raise ValueError(f"Request data is not a dictionary: {data}")
    return cast(Dict[str, Any], data)


class AddReturnTokenIds(CustomLogger):
    """Callback to add requests for return_token_ids to the request data."""

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> Optional[Union[Exception, str, Dict[str, Any]]]:
        try:
            data = _get_pre_call_data(args, kwargs)
        except Exception as e:
            return e

        # https://github.com/vllm-project/vllm/pull/22587
        return {**data, "return_token_ids": True}


class LightningSpanExporter(SpanExporter):

    def __init__(self, store: LightningStore):
        self.store = store

        self._buffer: List[ReadableSpan] = []
        self._lock = threading.RLock()

        # Single dedicated event loop running in a daemon thread.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="LightningSpanExporterLoop", daemon=True)
        self._loop_thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def shutdown(self) -> None:
        """Optional: call when your process exits."""
        try:

            def _stop():
                self._loop.stop()

            self._loop.call_soon_threadsafe(_stop)
            self._loop_thread.join(timeout=2.0)
            self._loop.close()
        except Exception:
            logger.exception("Error during exporter shutdown")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            for span in spans:
                self._buffer.append(span)

        # Run the async flush on our private loop, synchronously from caller’s POV.
        async def _locked_flush():
            with self._lock:
                return await self._maybe_flush()

        try:
            fut = asyncio.run_coroutine_threadsafe(_locked_flush(), self._loop)
            fut.result()  # bubble up any exceptions
        except Exception as e:
            logger.exception("Export flush failed: %s", e)
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    async def _maybe_flush(self):
        """Every span needs to find its parent span (or it's a root span) to be flushed.

        In a tree, we find "metadata.requester_custom_headers": "{'x-rollout-id': '123', 'x-attempt-id': '456', 'x-sequence-id': '1'}"
        to get the information we want to add to the store.
        """
        for root_span_id in self._get_root_span_ids():
            subtree_spans = self._pop_subtrees(root_span_id)
            if not subtree_spans:
                continue

            # Merge all custom headers in the subtree spans.
            # This is to find the rollout_id and attempt_id.
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
                try:
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
                logger.warning(f"No headers found in {len(subtree_spans)} subtree spans. Can't logging to store.")
                continue

            # Convert the rollout_id and attempt_id to str, sequence_id to int.
            rollout_id = headers_merged.get("x-rollout-id")
            attempt_id = headers_merged.get("x-attempt-id")
            sequence_id = headers_merged.get("x-sequence-id")
            if not rollout_id or not attempt_id or not sequence_id or not sequence_id.isdigit():
                logger.warning(
                    f"Missing or invalid rollout_id, attempt_id, or sequence_id in headers: {headers_merged}. Can't logging to store."
                )
                continue
            if not isinstance(rollout_id, str) or not isinstance(attempt_id, str):
                logger.warning(
                    f"rollout_id or attempt_id is not a string: {rollout_id}, {attempt_id}. Can't logging to store."
                )
                continue
            sequence_id_decimal = int(sequence_id)

            # Store the spans to the store.
            for span in subtree_spans:
                await self.store.add_otel_span(
                    rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id_decimal, readable_span=span
                )

    def _get_root_span_ids(self) -> Iterable[int]:
        for span in self._buffer:
            if span.parent is None:
                span_context = span.get_span_context()
                if span_context is not None:
                    yield span_context.span_id

    def _get_subtrees(self, root_span_id: int) -> Iterable[int]:
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
        """Get the subtree of a particular root span id and remove them from the buffer."""
        subtree_span_ids = set(self._get_subtrees(root_span_id))
        subtree_spans: List[ReadableSpan] = []
        new_buffer: List[ReadableSpan] = []
        for span in self._buffer:
            span_context = span.get_span_context()
            if span_context is not None and span_context.span_id in subtree_span_ids:
                subtree_spans.append(span)
            else:
                new_buffer.append(span)
        self._buffer = new_buffer
        return subtree_spans


class LightningOpenTelemetry(OpenTelemetry):
    """OpenTelemetry callback that logs the spans to lightning store.

    It gets a sequence id at the beginning where request is initiated so that
    they won't get mixed up due to the misaligned clock between the client node and the proxy node.

    It also stores every span to the lightning store so that we can use it for training.
    """

    def __init__(self, store: LightningStore):
        self.store = store
        config = OpenTelemetryConfig(exporter=LightningSpanExporter(store))
        super().__init__(config=config)  # pyright: ignore[reportUnknownMemberType]


class LLMProxy:

    def __init__(
        self, store: LightningStore, host: str, port: int, model_list: List[ModelConfig], litellm_config: Dict[str, Any]
    ):
        self.store = store
        self.host = host
        self.port = port
        self.model_list = model_list
        self.litellm_config = litellm_config
        self._server_thread = None
        self._config_file = None
        self._uvicorn_server = None
        self._ready_event = threading.Event()

    def initialize(self) -> None:
        # Add middleware here because it relies on "self.store".
        @app.middleware("http")
        async def rollout_attempt_middleware(request: Request, call_next):
            path = request.url.path

            match = re.match(r"^/rollout/([^/]+)/attempt/([^/]+)(/.*)?$", path)
            if match:
                rollout_id = match.group(1)
                attempt_id = match.group(2)
                new_path = match.group(3) if match.group(3) is not None else "/"

                request.scope["path"] = new_path
                request.scope["raw_path"] = new_path.encode()

                sequence_id = await self.store.get_next_span_sequence_id(rollout_id, attempt_id)

                request.scope["headers"] = list(request.scope["headers"]) + [
                    (b"x-rollout-id", rollout_id.encode()),
                    (b"x-attempt-id", attempt_id.encode()),
                    (b"x-sequence-id", str(sequence_id).encode()),
                ]

            response = await call_next(request)
            return response

        litellm.callbacks.extend(
            [  # pyright: ignore[reportUnknownMemberType]
                AddReturnTokenIds(),
                LightningOpenTelemetry(self.store),
            ]
        )

    def _wait_until_started(self, startup_timeout: float = 20.0):
        """Block until the uvicorn Server flips .started or we time out/exiting."""
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

    def start(self):
        self._config_file = tempfile.mktemp(suffix=".yaml")
        with open(self._config_file, "w") as fp:
            yaml.safe_dump(
                {
                    "model_list": self.model_list,
                    **self.litellm_config,
                },
                fp,
            )

        save_worker_config(config=self._config_file)

        self._uvicorn_server = uvicorn.Server(uvicorn.Config(app, host=self.host, port=self.port))

        def run_server():
            assert self._uvicorn_server is not None
            asyncio.run(self._uvicorn_server.serve())

        self._ready_event.clear()
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._wait_until_started()

    def stop(self):
        if self._config_file and os.path.exists(self._config_file):
            os.unlink(self._config_file)

        if self._server_thread is not None and self._uvicorn_server is not None and self._uvicorn_server.started:
            self._uvicorn_server.should_exit = True
            self._server_thread.join(timeout=20.0)  # Allow time for graceful shutdown.

    def as_resource(self, sampling_parameters: Dict[str, Any]) -> LLM:
        """
        Return an `LLM` Resource pointing at this proxy (OpenAI-compatible /v1).

        The `model` you pass at inference-time should be one of the model_name
        values from the constructed `model_list`. (e.g., 'backend-1', 'gpt-4o', ...).
        Clients will hit `endpoint/v1/...` like the OpenAI API. :contentReference[oaicite:4]{index=4}
        """
        return LLM(
            endpoint=f"http://{self.host}:{self.port}/v1",
            model="backend-1",  # default logical name; callers can override per-request
            sampling_parameters=dict(sampling_parameters or {}),
        )


def main():
    proxy = LLMProxy(
        store=InMemoryLightningStore(),
        host="0.0.0.0",
        port=9000,
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct",
                    "api_base": "http://127.0.0.1:8000/v1",
                },
            }
        ],
        litellm_config={},
    )

    proxy.initialize()
    proxy.start()

    import openai

    client = openai.OpenAI(base_url="http://127.0.0.1:9000/rollout/123/attempt/456", api_key="token-abc123")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        stream=False,
    )
    print(response)

    proxy.stop()


if __name__ == "__main__":
    main()
