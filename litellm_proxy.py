import asyncio
import os
import tempfile
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, TypedDict, Union

import litellm
import uvicorn
import yaml
from fastapi import Request
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.proxy.proxy_server import UserAPIKeyAuth, app, save_worker_config
from litellm.types.utils import ModelResponseStream


class ModelConfig(TypedDict):
    model_name: str
    litellm_params: Dict[str, Any]


class AddReturnTokenIds(CustomLogger):

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict[str, Any],
        call_type: Any,
    ) -> Optional[Union[Exception, str, dict]]:
        print(call_type, data)
        return {**data, "return_token_ids": True}


add_return_token_ids = AddReturnTokenIds()

litellm.callbacks.append(add_return_token_ids)

otel = OpenTelemetry()
litellm.callbacks.append(otel)


@app.middleware("http")
async def rollout_attempt_middleware(request: Request, call_next):
    path = request.url.path

    if path.startswith("/rollout/") and "/attempt/" in path:
        path_parts = path.split("/")
        if len(path_parts) >= 5 and path_parts[1] == "rollout" and path_parts[3] == "attempt":
            rollout_id = path_parts[2]
            attempt_id = path_parts[4]
            new_path = "/" + "/".join(path_parts[5:]) if len(path_parts) > 5 else "/"

            request.scope["path"] = new_path
            request.scope["raw_path"] = new_path.encode()

            request.scope["headers"] = list(request.scope["headers"]) + [
                (b"x-rollout-id", rollout_id.encode()),
                (b"x-attempt-id", attempt_id.encode()),
            ]

    response = await call_next(request)
    return response


class LLMProxy:

    def __init__(self, host: str, port: int, model_list: List[ModelConfig], litellm_config: Dict[str, Any]):
        self.host = host
        self.port = port
        self.model_list = model_list
        self.litellm_config = litellm_config
        self._server_thread = None
        self._config_file = None
        self._uvicorn_server = None
        self._ready_event = threading.Event()

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


def main():
    proxy = LLMProxy(
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

    proxy.start()

    import openai

    client = openai.OpenAI(base_url="http://127.0.0.1:9000/rollout/123/attempt/456", api_key="token-abc123")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)

    proxy.stop()


if __name__ == "__main__":
    main()
