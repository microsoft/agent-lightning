# Copyright (c) Microsoft. All rights reserved.

"""Programmatically launch and stop an vLLM server."""

import subprocess
import time
from contextlib import contextmanager
from typing import Optional

import httpx
from openai import OpenAI
from rich.console import Console

console = Console()


@contextmanager
def vllm_server(
    model_path: str,
    port: int,
    startup_timeout: float = 300.0,
    terminate_timeout: float = 10.0,
    gpu_memory_utilization: float = 0.7,
    auto_tool_choice: bool = True,
    tool_call_parser: Optional[str] = "hermes",
):
    """Serves a vLLM model from command line.

    Args:
        model_path: The path to the vLLM model. It can be either a local path or a Hugging Face model ID.
        port: The port to serve the model on.
        startup_timeout: The timeout for the server to start.
        terminate_timeout: The timeout for the server to terminate.
        gpu_memory_utilization: The GPU memory utilization for the server. Set it lower to avoid OOM.
        auto_tool_choice: Whether to enable auto tool choice.
        tool_call_parser: The tool call parser to use.
    """
    proc: Optional[subprocess.Popen[bytes]] = None
    try:
        vllm_serve_args = [
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--port",
            str(port),
        ]
        if auto_tool_choice:
            vllm_serve_args.append("--enable-auto-tool-choice")
        if tool_call_parser is not None:
            vllm_serve_args.append("--tool-call-parser")
            vllm_serve_args.append(tool_call_parser)

        proc = subprocess.Popen(["vllm", "serve", model_path, *vllm_serve_args])

        # Wait for the server to be ready
        url = f"http://localhost:{port}/health"
        start = time.time()
        client = httpx.Client()

        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                result = proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None
                time.sleep(0.5)
                if time.time() - start > startup_timeout:
                    raise RuntimeError(f"Server failed to start in {startup_timeout} seconds.") from None

        yield f"http://localhost:{port}/v1"
    finally:
        # Terminate the server
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(terminate_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    with vllm_server("Qwen/Qwen2.5-0.5B-Instruct", 8080) as endpoint:
        client = OpenAI(base_url=endpoint, api_key="dummy")
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[{"role": "user", "content": "Hello, what's your name?"}],
        )
        console.print(response)
        assert "qwen" in response.choices[0].message.content.lower()  # type: ignore
