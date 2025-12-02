import time
import httpx
import subprocess
from contextlib import contextmanager

@contextmanager
def vllm_server(
    model_path: str,
    port: int,
    startup_timeout: float = 300.0,
    terminate_timeout: float = 10.0,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.7,
):
    """Serves a vLLM model from command line.

    Args:
        model_path: The path to the vLLM model. It can be either a local path or a Hugging Face model ID.
        port: The port to serve the model on.
        startup_timeout: The timeout for the server to start.
        terminate_timeout: The timeout for the server to terminate.
        max_model_len: The maximum model length.
        gpu_memory_utilization: The GPU memory utilization for the server. Set it lower to avoid OOM.
        quantization: The quantization method.
        auto_tool_choice: Whether to enable auto tool choice.
        tool_call_parser: The tool call parser to use.
    """
    proc: Optional[subprocess.Popen[bytes]] = None
    try:
        vllm_serve_args = [
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--port",
            str(port),
        ]

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