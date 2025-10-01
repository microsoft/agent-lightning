# Copyright (c) Microsoft. All rights reserved.

import ast
import os
import socket
import subprocess
import sys
import time
from contextlib import closing
from typing import Any, Optional

import httpx
import openai
import pytest

from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.memory import InMemoryLightningStore

VLLM_AVAILABLE = False

try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils import FlexibleArgumentParser

    VLLM_AVAILABLE = True  # type: ignore
except ImportError:
    AsyncEngineArgs = None
    get_model_loader = None
    FlexibleArgumentParser = None
    ServeSubcommand = None

# TODO: mark the whole module to skip if no GPU is available


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class RemoteOpenAIServer:
    """
    A context manager for launching and interacting with a remote vLLM-based
    OpenAI-compatible server instance.

    This class handles:
      - Preparing the environment and spawning the vLLM server process
      - Ensuring that the requested model is downloaded before server startup
      - Polling and health-checking the server until it is ready
      - Providing helper methods to construct URLs for API calls
      - Returning configured synchronous and asynchronous OpenAI clients
        that can communicate with the launched server

    Typical usage:
        with RemoteOpenAIServer(vllm_serve_args, port, model) as server:
            client = server.get_client()
            response = client.chat.completions.create(...)

    Attributes:
        DUMMY_API_KEY (str): A placeholder API key for compatibility
                             (vLLM does not require authentication).
        host (str): Host address of the server (default: "localhost").
        port (int): TCP port number for the server.
        proc (subprocess.Popen): Handle to the launched server process.
    """

    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def _start_server(self, model: str, vllm_serve_args: list[str], env_dict: Optional[dict[str, str]]) -> None:
        """Subclasses override this method to customize server process launch"""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # safer CUDA init
        if env_dict is not None:
            env.update(env_dict)
        self.proc: subprocess.Popen[bytes] = subprocess.Popen(
            ["vllm", "serve", model, *vllm_serve_args],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],  # should not include the model name
        env_dict: Optional[dict[str, str]] = None,
        seed: Optional[int] = 0,
        max_wait_seconds: Optional[float] = None,
    ) -> None:
        if (
            not VLLM_AVAILABLE
            or AsyncEngineArgs is None
            or get_model_loader is None
            or FlexibleArgumentParser is None
            or ServeSubcommand is None
        ):
            raise ImportError("vLLM is not available")

        self.model = model

        parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        parser = ServeSubcommand().subparser_init(subparsers)  # pyright: ignore[reportUnknownMemberType]
        args = parser.parse_args(["--model", model, *vllm_serve_args])
        assert args is not None
        self.host = str(args.host or "localhost")
        self.port = int(args.port)

        # download the model before starting the server to avoid timeout
        is_local = os.path.isdir(model)
        if not is_local:
            engine_args = AsyncEngineArgs.from_cli_args(args)
            model_config = engine_args.create_model_config()
            load_config = engine_args.create_load_config()

            model_loader = get_model_loader(load_config)
            model_loader.download_model(model_config)

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 240
        self._wait_for_server(url=self.url_for("health"), timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def _poll(self) -> Optional[int]:
        """Subclasses override this method to customize process polling"""
        return self.proc.poll()

    def _wait_for_server(self, *, url: str, timeout: float):
        start = time.time()
        client = httpx.Client()

        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None
                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs: Any):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs: Any):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )


@pytest.fixture(scope="module")
def qwen25_model():
    with RemoteOpenAIServer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        vllm_serve_args=[
            "--gpu-memory-utilization",
            "0.8",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--port",
            str(_get_free_port()),
        ],
    ) as server:
        yield server


def test_qwen25_model_sanity(qwen25_model: RemoteOpenAIServer):
    client = qwen25_model.get_client()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "Hello, world!"}],
        stream=False,
    )
    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_basic_integration(qwen25_model: RemoteOpenAIServer):
    store = InMemoryLightningStore()
    proxy = LLMProxy(
        port=_get_free_port(),
        model_list=[
            {
                "model_name": "gpt-4o-arbitrary",
                "litellm_params": {
                    "model": "hosted_vllm/" + qwen25_model.model,
                    "api_base": qwen25_model.url_for("v1"),
                },
            }
        ],
        store=store,
    )

    rollout = await store.start_rollout(None)

    proxy.initialize()
    proxy.start()

    resource = proxy.as_resource(rollout.rollout_id, rollout.attempt.attempt_id)

    import openai

    client = openai.OpenAI(base_url=resource.endpoint, api_key="token-abc123")
    response = client.chat.completions.create(
        model="gpt-4o-arbitrary",
        messages=[{"role": "user", "content": "Repeat after me: Hello, world!"}],
        stream=False,
    )
    assert response.choices[0].message.content is not None
    assert "Hello, world" in response.choices[0].message.content

    proxy.stop()

    spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)

    # Verify all spans have correct rollout_id, attempt_id, and sequence_id
    assert len(spans) > 0, "Should have captured spans"
    for span in spans:
        assert span.rollout_id == rollout.rollout_id, f"Span {span.name} has incorrect rollout_id"
        assert span.attempt_id == rollout.attempt.attempt_id, f"Span {span.name} has incorrect attempt_id"
        assert span.sequence_id == 1, f"Span {span.name} has incorrect sequence_id"

    # Find the raw_gen_ai_request span and verify token IDs
    raw_gen_ai_spans = [s for s in spans if s.name == "raw_gen_ai_request"]
    assert len(raw_gen_ai_spans) == 1, f"Expected 1 raw_gen_ai_request span, found {len(raw_gen_ai_spans)}"
    raw_span = raw_gen_ai_spans[0]

    # Verify prompt_token_ids is present and non-empty
    assert (
        "llm.hosted_vllm.prompt_token_ids" in raw_span.attributes
    ), "prompt_token_ids not found in raw_gen_ai_request span"
    prompt_token_ids: list[int] = ast.literal_eval(raw_span.attributes["llm.hosted_vllm.prompt_token_ids"])  # type: ignore
    assert isinstance(prompt_token_ids, list), "prompt_token_ids should be a list"
    assert len(prompt_token_ids) > 0, "prompt_token_ids should not be empty"
    assert all(isinstance(tid, int) for tid in prompt_token_ids), "All prompt token IDs should be integers"

    # Verify response token_ids is present in choices
    assert "llm.hosted_vllm.choices" in raw_span.attributes, "choices not found in raw_gen_ai_request span"
    choices: list[dict[str, Any]] = ast.literal_eval(raw_span.attributes["llm.hosted_vllm.choices"])  # type: ignore
    assert len(choices) > 0, "Should have at least one choice"
    assert "token_ids" in choices[0], "token_ids not found in choice"
    response_token_ids: list[int] = choices[0]["token_ids"]
    assert isinstance(response_token_ids, list), "response token_ids should be a list"
    assert len(response_token_ids) > 0, "response token_ids should not be empty"
    assert all(isinstance(tid, int) for tid in response_token_ids), "All response token IDs should be integers"

    # Find the litellm_request span and verify gen_ai prompts/completions
    litellm_spans = [s for s in spans if s.name == "litellm_request"]
    assert len(litellm_spans) == 1, f"Expected 1 litellm_request span, found {len(litellm_spans)}"
    litellm_span = litellm_spans[0]

    # Verify gen_ai.prompt attributes
    assert "gen_ai.prompt.0.role" in litellm_span.attributes, "gen_ai.prompt.0.role not found"
    assert litellm_span.attributes["gen_ai.prompt.0.role"] == "user", "Expected user role in prompt"
    assert "gen_ai.prompt.0.content" in litellm_span.attributes, "gen_ai.prompt.0.content not found"
    assert litellm_span.attributes["gen_ai.prompt.0.content"] == "Repeat after me: Hello, world!"

    # Verify gen_ai.completion attributes
    assert "gen_ai.completion.0.role" in litellm_span.attributes, "gen_ai.completion.0.role not found"
    assert litellm_span.attributes["gen_ai.completion.0.role"] == "assistant", "Expected assistant role in completion"
    assert "gen_ai.completion.0.content" in litellm_span.attributes, "gen_ai.completion.0.content not found"
    assert "gen_ai.completion.0.finish_reason" in litellm_span.attributes, "gen_ai.completion.0.finish_reason not found"
