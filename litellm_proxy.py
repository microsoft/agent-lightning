import asyncio
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, TypedDict

import uvicorn
import yaml
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.proxy.proxy_server import app, save_worker_config

otel = OpenTelemetry()


class ModelConfig(TypedDict):
    model_name: str
    litellm_params: Dict[str, Any]


from typing import Any, AsyncGenerator, Literal, Optional, Union

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import UserAPIKeyAuth
from litellm.types.utils import ModelResponseStream


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger):  # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self):
        pass

    #### CALL HOOKS - proxy only ####

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[
        Union[Exception, str, dict]
    ]:  # raise exception if invalid, return a str for the user to receive - if rejected, or return a modified dictionary for passing into litellm
        print(call_type, data)
        return {**data, "return_token_ids": True}


proxy_handler_instance = MyCustomHandler()

litellm.callbacks.append(MyCustomHandler())
litellm.callbacks.append(otel)


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

    client = openai.OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="token-abc123")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)

    proxy.stop()


if __name__ == "__main__":
    main()
