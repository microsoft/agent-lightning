# Copyright (c) Microsoft. All rights reserved.

import os
import subprocess
import sys
import time
from typing import Any, Optional

import httpx
import openai

VLLM_AVAILABLE = False
VLLM_UNAVAILABLE_REASON = ""

try:
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils import FlexibleArgumentParser

    VLLM_AVAILABLE = True  # type: ignore
    VLLM_VERSION = tuple(int(v) for v in vllm.__version__.split("."))
except ImportError as e:
    AsyncEngineArgs = None
    get_model_loader = None
    FlexibleArgumentParser = None
    ServeSubcommand = None
    VLLM_VERSION = (0, 0, 0)  # type: ignore
    VLLM_UNAVAILABLE_REASON = str(e)  # type: ignore


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

        if VLLM_VERSION >= (0, 10, 2):
            # Supports return_token_ids
            self.proc: subprocess.Popen[bytes] = subprocess.Popen(
                ["vllm", "serve", model, *vllm_serve_args],
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        else:
            # Does not support return_token_ids
            self.proc = subprocess.Popen(
                ["python", "-m", "agentlightning.cli.vllm", "serve", model, *vllm_serve_args],
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
            raise ImportError("vLLM is not available: " + VLLM_UNAVAILABLE_REASON)

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
